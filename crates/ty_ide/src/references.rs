//! This module implements the core functionality of the "references",
//! "document highlight" and "rename" language server features. It locates
//! all references to a named symbol. Unlike a simple text search for the
//! symbol's name, this is a "semantic search" where the text and the semantic
//! meaning must match.
//!
//! Some symbols (such as parameters and local variables) are visible only
//! within their scope. All other symbols, such as those defined at the global
//! scope or within classes, are visible outside the module. Finding
//! all references to these externally-visible symbols therefore requires
//! an expensive search of all source files in the workspace.

use std::collections::hash_map::Entry;

use crate::goto::{Definitions, GotoTarget};
use crate::{Db, FxIndexSet, ReferenceKind, ReferenceTarget};
use rayon::prelude::*;
use ruff_db::parsed::{ParsedModuleRef, parsed_module};
use ruff_python_ast::find_node::{CoveringNode, covering_node};
use ruff_python_ast::token::Tokens;
use ruff_python_ast::{
    self as ast, AnyNodeRef,
    visitor::source_order::{SourceOrderVisitor, TraversalSignal},
};
use ruff_text_size::Ranged;
use rustc_hash::{FxHashMap, FxHashSet};
use ty_module_resolver::{ImportingFile, file_to_module, resolve_module};
use ty_project::parallel::{ParallelIteratorExt, minimum_parallel_job_len};
use ty_python_core::definition::{Definition, DefinitionKind, DefinitionState};
use ty_python_core::scope::{FileScopeId, NodeWithScopeKind, ScopeKind};
use ty_python_core::{ProgramFile, place_table};
use ty_python_semantic::{
    ImportAliasResolution, ResolvedDefinition, SemanticModel, contains_identifier,
    definitions_for_imported_symbol, fixture_reference_candidates, fixture_reference_identities,
    fixture_reference_search_files,
};

/// Salsa snapshots coordinate clone and drop through shared state. For cached files that don't
/// contain the target, that coordination can cost more than the file scan and scales poorly when
/// many short-lived jobs finish concurrently. A 64-file minimum on large projects amortizes the
/// task and snapshot overhead. Smaller projects lower the minimum to retain enough work for
/// stealing.
const MAX_MIN_FILES_PER_PARALLEL_JOB: usize = 64;

/// Mode for references search behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferencesMode {
    /// Find all references including the declaration
    References,
    /// Find all references but skip the declaration
    ReferencesSkipDeclaration,
    /// Find references for rename operations, limited to current file only
    Rename,
    /// Find references for multi-file rename operations (searches across all files)
    RenameMultiFile,
    /// Find references for document highlights (limits search to current file)
    DocumentHighlights,
}

impl ReferencesMode {
    pub(super) fn to_import_alias_resolution(self) -> ImportAliasResolution {
        match self {
            // Resolve import aliases for find references:
            // ```py
            // from warnings import deprecated as my_deprecated
            //
            // @my_deprecated
            // def foo
            // ```
            //
            // When finding references on `my_deprecated`, we want to find all usages of `deprecated` across the entire
            // project.
            Self::References | Self::ReferencesSkipDeclaration => {
                ImportAliasResolution::ResolveAliases
            }
            // For rename, don't resolve import aliases.
            //
            // ```py
            // from warnings import deprecated as my_deprecated
            //
            // @my_deprecated
            // def foo
            // ```
            // When renaming `my_deprecated`, only rename the alias, but not the original definition in `warnings`.
            Self::Rename | Self::RenameMultiFile | Self::DocumentHighlights => {
                ImportAliasResolution::PreserveAliases
            }
        }
    }
}

/// Find all references to a symbol at the given position.
/// Search for references across all files in the project.
pub(crate) fn references(
    db: &dyn Db,
    file: ProgramFile<'_>,
    goto_target: &GotoTarget,
    mode: ReferencesMode,
) -> Option<Vec<ReferenceTarget>> {
    let source_file = file.file(db);
    let model = SemanticModel::new(db, file);
    let mut target_definitions =
        goto_target.definitions(&model, mode.to_import_alias_resolution())?;
    let mut fixture_references = Vec::new();

    if matches!(
        mode,
        ReferencesMode::References | ReferencesMode::ReferencesSkipDeclaration
    ) && goto_target_has_fixture_identities(db, &model, goto_target)
    {
        let (fixture_identities, ordinary_definitions) =
            partition_by_fixture_identity(db, &target_definitions);

        if !fixture_identities.is_empty() {
            if ordinary_definitions.is_empty() {
                let fixture_references =
                    pytest_fixture_references(db, file, &fixture_identities, mode);
                return (!fixture_references.is_empty()).then_some(fixture_references);
            }

            target_definitions = Definitions::new(ordinary_definitions);
            fixture_references = pytest_fixture_references(db, file, &fixture_identities, mode);
        }
    }

    let is_externally_visible_symbol =
        has_any_external_visible_definitions(db, &target_definitions);
    let target_definitions = target_definitions.goto_declaration(&model, goto_target)?;

    // Check if we should search across files based on the mode
    let search_across_files = matches!(
        mode,
        ReferencesMode::References
            | ReferencesMode::ReferencesSkipDeclaration
            | ReferencesMode::RenameMultiFile
    );

    // Parameters are local by scope, but they can have cross-file references via keyword
    // argument labels (e.g. `f(param=...)`). Handle this case with a narrow scan that only
    // considers keyword arguments.
    let is_parameter = parameter_owner_is_externally_visible(db, &target_definitions);

    let searches = LocalReferenceSearches::single(
        goto_target.to_string()?.into_owned(),
        target_definitions,
        mode.to_import_alias_resolution(),
    );

    // Find all of the references to the symbol within this file.
    let mut references = references_for_file(db, file, &searches, mode);

    if search_across_files && (is_parameter || is_externally_visible_symbol) {
        let program = model.program();
        let files = db.project().files(db);
        let files: Vec<_> = files
            .iter()
            .copied()
            .filter(|other| *other != source_file)
            .collect();
        let minimum_job_len = minimum_parallel_job_len(files.len(), MAX_MIN_FILES_PER_PARALLEL_JOB);
        let other_references = files
            .into_par_iter()
            .with_min_len(minimum_job_len)
            .map_with_db(db, |db, other_file| {
                let source = ruff_db::source::source_text(db, other_file);
                if !searches.may_match_source(&source) {
                    return Vec::new();
                }

                let other_file = ProgramFile::new(db, other_file, program);

                if is_externally_visible_symbol {
                    references_for_file(db, other_file, &searches, mode)
                } else {
                    references_for_keyword_arguments_in_file(db, other_file, &searches, mode)
                }
            })
            .flat_map_iter(|references| references)
            .collect::<Vec<_>>();

        references.extend(other_references);
    }

    if !fixture_references.is_empty() {
        references.extend(fixture_references);
        let mut seen_ranges = FxHashSet::default();
        references.retain(|reference| seen_ranges.insert(reference.file_range()));
    }

    if references.is_empty() {
        None
    } else {
        Some(references)
    }
}

/// Whether or not the goto target is related to pytest fixture declaration.
///
/// This is true when a parameter (implicitly) requests a pytest fixture, when
/// a function itself defines a fixture, or when an import or stub definition
/// references a fixture.
fn goto_target_has_fixture_identities(
    db: &dyn Db,
    model: &SemanticModel<'_>,
    goto_target: &GotoTarget<'_>,
) -> bool {
    goto_target
        .definitions(model, ImportAliasResolution::PreserveAliases)
        .is_some_and(|definitions| {
            definitions.iter().any(|resolved| {
                resolved.definition().is_some_and(|definition| {
                    !fixture_reference_identities(db, definition).is_empty()
                })
            })
        })
}

/// Split the given list of definitions into two lists:
///
/// 1. All the resolved pytest fixture identities from the definitions in the original list
/// 2. All of the ordinary (i.e., non-fixture-related) definitions from the original list.
fn partition_by_fixture_identity<'db>(
    db: &'db dyn Db,
    definitions: &Definitions<'db>,
) -> (Vec<Definition<'db>>, Vec<ResolvedDefinition<'db>>) {
    let mut seen = FxHashSet::default();
    let mut fixture_identities = Vec::new();
    let mut ordinary_definitions = Vec::new();

    for resolved in definitions {
        let Some(definition) = resolved.definition() else {
            ordinary_definitions.push(resolved.clone());
            continue;
        };

        let identities = fixture_reference_identities(db, definition);

        if identities.is_empty() {
            ordinary_definitions.push(resolved.clone());
        } else {
            fixture_identities.extend(
                identities
                    .iter()
                    .copied()
                    .filter(|target| seen.insert(*target)),
            );
        }
    }

    (fixture_identities, ordinary_definitions)
}

/// Finds all references related to any of the given pytest fixture identities.
fn pytest_fixture_references(
    db: &dyn Db,
    source_file: ProgramFile<'_>,
    fixture_identities: &[Definition<'_>],
    mode: ReferencesMode,
) -> Vec<ReferenceTarget> {
    let program = source_file.program(db);

    // Search the source file, all project files, the files that define the fixtures (and their
    // stubs), and installed plugin files. A file can belong to more than one group, so deduplicate
    // the combined list before scanning it.
    let files: FxIndexSet<_> = std::iter::once(source_file.file(db))
        .chain(db.project().files(db).iter().copied())
        .chain(fixture_identities.iter().flat_map(|definition| {
            let target_file = definition.program_file(db);
            std::iter::once(target_file.file(db)).chain(
                fixture_identity_stub_file(db, *definition).map(|stub_file| stub_file.file(db)),
            )
        }))
        .chain(
            fixture_reference_search_files(db, program)
                .iter()
                .map(|file| file.file(db)),
        )
        .collect();
    let files: Vec<_> = files
        .into_iter()
        .map(|file| ProgramFile::new(db, file, program))
        .collect();

    let fixture_identities = fixture_identities.to_vec();

    // Use the same batched Salsa-snapshot strategy as ordinary cross-file search.
    let minimum_job_len = minimum_parallel_job_len(files.len(), MAX_MIN_FILES_PER_PARALLEL_JOB);
    let references = files
        .into_par_iter()
        .with_min_len(minimum_job_len)
        .map_with_db(db, |db, file| {
            pytest_fixture_references_for_file(db, file, &fixture_identities, mode)
        })
        .flat_map_iter(|references| references)
        .collect::<Vec<_>>();

    let mut seen_ranges = FxHashSet::default();
    references
        .into_iter()
        .filter(|reference| seen_ranges.insert(reference.file_range()))
        .collect()
}

/// Finds references in `file` related to any of the given fixture identities.
fn pytest_fixture_references_for_file<'db>(
    db: &'db dyn Db,
    file: ProgramFile<'db>,
    fixture_identities: &[Definition<'db>],
    mode: ReferencesMode,
) -> Vec<ReferenceTarget> {
    let source = ruff_db::source::source_text(db, file.file(db));
    let candidates = fixture_reference_candidates(db, file);
    let parsed = parsed_module(db, file.python_file(db));
    let module = parsed.load(db);
    let model = SemanticModel::new(db, file);

    // Represent every matching fixture declaration, exposure, and request as an ordinary
    // name-based reference search. This preserves each binding's local name and alias behavior
    // while reusing the existing semantic reference walker.
    let mut searches = LocalReferenceSearches::default();

    for fixture_identity in fixture_identities {
        let Some(name) = fixture_identity.name(db) else {
            continue;
        };
        searches.insert(
            name,
            ResolvedDefinition::Definition(*fixture_identity),
            mode.to_import_alias_resolution(),
        );
    }

    // An ambiguous fixture request can connect multiple fixture identities:
    //
    // ```py
    // if flag:
    //     from first import first as resource
    // else:
    //     from second import second as resource
    //
    // def test_use(resource): ...
    // ```
    //
    // A search starting from `first` should include the request and its uses because the request
    // may resolve to `first`, but it should not include the `second as resource` declaration.
    //
    // Record non-matching fixture candidates so that we can provide that behaviour.
    let mut non_matching_fixture_candidate_ranges = Vec::new();

    for &candidate in candidates {
        let candidate_fixture_identities = fixture_reference_identities(db, candidate);
        let shares_fixture_identity = candidate_fixture_identities
            .iter()
            .any(|fixture_identity| fixture_identities.contains(fixture_identity));

        if !shares_fixture_identity {
            if !candidate_fixture_identities.is_empty() {
                non_matching_fixture_candidate_ranges
                    .push(candidate.focus_range(db, &module).range());
            }
            continue;
        }

        if let DefinitionKind::ImportFrom(import) = candidate.kind(db) {
            let import_node = import.import(&module);
            let alias = import.alias(&module);

            // An imported name can resolve to a stub or a renamed re-export instead of the runtime
            // fixture definition. For example:
            //
            // ```python
            // # reexports.py
            // from fixtures import resource as middle
            //
            // # test_example.py
            // from reexports import middle
            // ```
            //
            // While scanning `test_example.py`, add a search for the name `middle` that targets
            // the `middle` import definition in `reexports.py`. Each file adds such searches for
            // its own imports, so intermediate definitions are not searched in unrelated files.
            for resolved in definitions_for_imported_symbol(
                &model,
                import_node,
                alias.name.as_str(),
                ImportAliasResolution::PreserveAliases,
            ) {
                let Some(definition) = resolved.definition() else {
                    continue;
                };
                if fixture_identities.contains(&definition)
                    || !fixture_reference_identities(db, definition)
                        .iter()
                        .any(|identity| fixture_identities.contains(identity))
                {
                    continue;
                }
                searches.insert(
                    alias.name.to_string(),
                    ResolvedDefinition::Definition(definition),
                    ImportAliasResolution::PreserveAliases,
                );
            }
        }

        let is_parameter = matches!(candidate.kind(db), DefinitionKind::Parameter(_));
        let import_alias_resolution = if is_parameter {
            mode.to_import_alias_resolution()
        } else {
            // An exposure participates through its local binding. Resolving its alias here could
            // include a definition for a different fixture identity when the binding is ambiguous.
            ImportAliasResolution::PreserveAliases
        };
        let Some(symbol) = candidate.place(db).as_symbol() else {
            continue;
        };
        let name = place_table(db, candidate.scope(db))
            .symbol(symbol)
            .name()
            .to_string();

        searches.insert(
            name,
            ResolvedDefinition::Definition(candidate),
            import_alias_resolution,
        );
    }

    let mut references = if searches.may_match_source(&source) {
        references_for_parsed_file(&model, &module, &searches, mode)
    } else {
        Vec::new()
    };

    references.retain(|reference| {
        !non_matching_fixture_candidate_ranges
            .iter()
            .any(|range| range.contains_range(reference.range()))
    });

    references
}

/// Semantic searches to perform during a single traversal of a file.
///
/// An ordinary reference search usually has one name:
///
/// ```python
/// resource = 1
/// print(resource)
/// ```
///
/// A pytest fixture reference search can have several names for the same fixture family:
///
/// ```python
/// from fixtures import resource as alias
///
/// def test_use(alias): ...
/// ```
///
/// This fixture search contains both `resource`, for the imported name, and
/// `alias`, for the local binding and fixture request.
///
/// Storing single-name and multi-name searches separately lets the common
/// single-name case use a direct string comparison. Searches for several names
/// use a hash lookup so they can share one syntax-tree walk.
#[derive(Default)]
enum LocalReferenceSearches<'db> {
    /// No names have been added yet.
    #[default]
    Empty,
    /// Exactly one identifier name is part of the search.
    Single {
        /// The identifier text to look for in the syntax tree.
        name: String,
        /// The semantic definitions and alias-resolution behavior associated with `name`.
        targets: ReferenceTargets<'db>,
    },
    /// Two or more identifier names are part of the search.
    Multiple {
        /// Maps each identifier text to its semantic definitions and alias-resolution behavior.
        by_name: FxHashMap<String, ReferenceTargets<'db>>,
    },
}

impl<'db> LocalReferenceSearches<'db> {
    /// Creates the common single-name search without allocating a hash table.
    fn single(
        name: String,
        definitions: Definitions<'db>,
        import_alias_resolution: ImportAliasResolution,
    ) -> Self {
        Self::Single {
            name,
            targets: ReferenceTargets::new(definitions, import_alias_resolution),
        }
    }

    /// Adds one semantic definition under `name` and the given alias-resolution behavior.
    fn insert(
        &mut self,
        name: String,
        definition: ResolvedDefinition<'db>,
        import_alias_resolution: ImportAliasResolution,
    ) {
        // Take ownership of the current state so that a second distinct name
        // can promote `Single` to `Multiple` without cloning either the first
        // name or its definitions.
        let current = std::mem::take(self);

        *self = match current {
            Self::Empty => Self::Single {
                name,
                targets: ReferenceTargets::from_definition(definition, import_alias_resolution),
            },
            Self::Single {
                name: existing_name,
                mut targets,
            } if existing_name == name => {
                // One name can need both alias-resolution behaviors (see the
                // comment on `ReferenceTargets`), so merge by behavior rather
                // than creating another name entry.
                targets.insert(definition, import_alias_resolution);
                Self::Single {
                    name: existing_name,
                    targets,
                }
            }
            Self::Single {
                name: existing_name,
                targets: existing_targets,
            } => {
                // Adding a second identifier name switches subsequent lookups to the hash table.
                let mut by_name = FxHashMap::default();
                by_name.insert(existing_name, existing_targets);
                by_name.insert(
                    name,
                    ReferenceTargets::from_definition(definition, import_alias_resolution),
                );
                Self::Multiple { by_name }
            }
            Self::Multiple { mut by_name } => {
                match by_name.entry(name) {
                    Entry::Occupied(mut entry) => {
                        entry.get_mut().insert(definition, import_alias_resolution);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(ReferenceTargets::from_definition(
                            definition,
                            import_alias_resolution,
                        ));
                    }
                }
                Self::Multiple { by_name }
            }
        };
    }

    /// Returns the semantic targets associated with the exact identifier text `name`.
    fn for_name(&self, name: &str) -> Option<&ReferenceTargets<'db>> {
        match self {
            Self::Empty => None,
            Self::Single {
                name: target_name,
                targets,
            } => (target_name == name).then_some(targets),
            Self::Multiple { by_name } => by_name.get(name),
        }
    }

    /// Returns whether `source` contains at least one identifier name in this search.
    fn may_match_source(&self, source: &str) -> bool {
        match self {
            Self::Empty => false,
            Self::Single { name, .. } => contains_identifier(source, name),
            Self::Multiple { by_name } => {
                by_name.keys().any(|name| contains_identifier(source, name))
            }
        }
    }
}

/// Semantic targets for one identifier name, separated by import-alias behavior.
///
/// The same name can need both behaviors:
///
/// ```python
/// # reexports.py
/// from fixtures import resource as middle
///
/// # test_example.py
/// from reexports import middle
///
/// def test_use(middle): ...
/// ```
///
/// In `test_example.py`, the imported `middle` must match the `middle` definition
/// exposed by `reexports.py`. The fixture parameter also has the name `middle`,
/// but it must resolve through that import to the original `resource` fixture.
struct ReferenceTargets<'db> {
    /// Definitions matched after following import aliases, such as the fixture
    /// requested by the `middle` parameter above.
    resolve_aliases: Option<Definitions<'db>>,
    /// Definitions matched while keeping import bindings distinct, such as the
    /// `middle` imported from `reexports.py` above.
    preserve_aliases: Option<Definitions<'db>>,
}

impl<'db> ReferenceTargets<'db> {
    /// Creates targets for one alias-resolution behavior from an existing definition set.
    fn new(definitions: Definitions<'db>, import_alias_resolution: ImportAliasResolution) -> Self {
        match import_alias_resolution {
            ImportAliasResolution::ResolveAliases => Self {
                resolve_aliases: Some(definitions),
                preserve_aliases: None,
            },
            ImportAliasResolution::PreserveAliases => Self {
                resolve_aliases: None,
                preserve_aliases: Some(definitions),
            },
        }
    }

    /// Creates targets for one alias-resolution behavior from a single definition.
    fn from_definition(
        definition: ResolvedDefinition<'db>,
        import_alias_resolution: ImportAliasResolution,
    ) -> Self {
        Self::new(Definitions::new(vec![definition]), import_alias_resolution)
    }

    /// Adds a definition to the set for the given alias-resolution behavior.
    fn insert(
        &mut self,
        definition: ResolvedDefinition<'db>,
        import_alias_resolution: ImportAliasResolution,
    ) {
        // Definitions using the same behavior can share one intersection check at each matching
        // identifier. The other slot remains available if the same name also needs that behavior.
        let definitions = match import_alias_resolution {
            ImportAliasResolution::ResolveAliases => &mut self.resolve_aliases,
            ImportAliasResolution::PreserveAliases => &mut self.preserve_aliases,
        };
        if let Some(definitions) = definitions {
            definitions.insert(definition);
        } else {
            *definitions = Some(Definitions::new(vec![definition]));
        }
    }

    /// Iterates over each populated alias-resolution behavior and its definitions.
    fn iter(&self) -> impl Iterator<Item = (ImportAliasResolution, &Definitions<'db>)> + '_ {
        [
            self.resolve_aliases
                .as_ref()
                .map(|definitions| (ImportAliasResolution::ResolveAliases, definitions)),
            self.preserve_aliases
                .as_ref()
                .map(|definitions| (ImportAliasResolution::PreserveAliases, definitions)),
        ]
        .into_iter()
        .flatten()
    }

    /// Iterates over all semantic definitions, regardless of alias-resolution behavior.
    fn definitions(&self) -> impl Iterator<Item = &ResolvedDefinition<'db>> + '_ {
        self.iter().flat_map(|(_, definitions)| definitions.iter())
    }
}

/// Returns the stub file through which a fixture identity may be exposed.
fn fixture_identity_stub_file<'db>(
    db: &'db dyn Db,
    target: Definition<'db>,
) -> Option<ProgramFile<'db>> {
    let target_file = target.program_file(db);
    let resolver_file = target_file.resolver_file(db);
    let target_module = file_to_module(db, resolver_file)?;
    let stub_module = resolve_module(
        db,
        ImportingFile::ResolverFile(resolver_file),
        target_module.name(db),
    )?;
    let stub_file = stub_module.file(db)?;

    stub_file
        .is_stub(db)
        .then(|| ProgramFile::new(db, stub_file, target_file.program(db)))
}

fn references_for_keyword_arguments_in_file(
    db: &dyn Db,
    file: ProgramFile<'_>,
    searches: &LocalReferenceSearches<'_>,
    mode: ReferencesMode,
) -> Vec<ReferenceTarget> {
    // This path is used for cross-file parameter keyword-label references.
    // DocumentHighlights is same-file-only and should never route through here.
    debug_assert!(
        !matches!(mode, ReferencesMode::DocumentHighlights),
        "keyword-label cross-file scan should not run in DocumentHighlights mode"
    );

    let parsed = parsed_module(db, file.python_file(db));
    let module = parsed.load(db);
    let model = SemanticModel::new(db, file);
    let mut references = Vec::new();

    let mut finder = KeywordArgumentReferencesFinder(LocalReferencesFinder {
        model: &model,
        tokens: module.tokens(),
        searches,
        references: &mut references,
        mode,
        ancestors: Vec::new(),
    });

    AnyNodeRef::from(module.syntax()).visit_source_order(&mut finder);

    references
}

/// Returns whether `node` assigns `value` to the sole target `__slots__`, e.g.
/// `__slots__ = (...)` or `__slots__: tuple = (...)`.
fn is_slots_assignment(node: AnyNodeRef<'_>, value: AnyNodeRef<'_>) -> bool {
    match node {
        AnyNodeRef::StmtAssign(assign) => {
            assign.value.range() == value.range()
                && matches!(
                    assign.targets.as_slice(),
                    [ast::Expr::Name(name)] if name.id.as_str() == "__slots__"
                )
        }
        AnyNodeRef::StmtAnnAssign(assign) => {
            assign
                .value
                .as_deref()
                .is_some_and(|assigned| assigned.range() == value.range())
                && matches!(
                    assign.target.as_ref(),
                    ast::Expr::Name(name) if name.id.as_str() == "__slots__"
                )
        }
        _ => false,
    }
}

/// Find all references to a local symbol within the current file.
/// The behavior depends on the provided mode.
fn references_for_file(
    db: &dyn Db,
    file: ProgramFile<'_>,
    searches: &LocalReferenceSearches<'_>,
    mode: ReferencesMode,
) -> Vec<ReferenceTarget> {
    let parsed = parsed_module(db, file.python_file(db));
    let module = parsed.load(db);
    let model = SemanticModel::new(db, file);

    references_for_parsed_file(&model, &module, searches, mode)
}

fn references_for_parsed_file<'a>(
    model: &'a SemanticModel<'a>,
    module: &'a ParsedModuleRef,
    searches: &'a LocalReferenceSearches<'a>,
    mode: ReferencesMode,
) -> Vec<ReferenceTarget> {
    let mut references = Vec::new();

    let mut finder = LocalReferencesFinder {
        model,
        searches,
        references: &mut references,
        mode,
        tokens: module.tokens(),
        ancestors: Vec::new(),
    };

    AnyNodeRef::from(module.syntax()).visit_source_order(&mut finder);

    references
}

/// Determines whether the resolved definitions can have references outside their file.
pub(crate) fn has_any_external_visible_definitions(
    db: &dyn Db,
    definitions: &Definitions<'_>,
) -> bool {
    definitions.iter().any(|definition| match definition {
        ResolvedDefinition::Definition(definition) => match definition.scope(db).scope(db).kind() {
            ScopeKind::Module | ScopeKind::Class => true,
            ScopeKind::Comprehension => {
                matches!(definition.kind(db), DefinitionKind::NamedExpression(_))
                    && definition.place(db).as_symbol().is_some_and(|symbol_id| {
                        ty_python_core::semantic_index(db, definition.program_file(db))
                            .symbol_resolves_to_global_scope(symbol_id, definition.file_scope(db))
                    })
            }
            ScopeKind::TypeParams
            | ScopeKind::Function
            | ScopeKind::Lambda
            | ScopeKind::TypeAlias => false,
        },
        ResolvedDefinition::Module(_) | ResolvedDefinition::FileWithRange(_) => true,
    })
}

/// Determine whether a parameter's owning callable is externally visible.
///
/// Parameters are local by scope, but their keyword-argument labels can appear across files
/// when the owning callable is visible outside of the current module.
fn parameter_owner_is_externally_visible(
    db: &dyn Db,
    target_definitions: &Definitions<'_>,
) -> bool {
    target_definitions
        .iter()
        .any(|target| parameter_owner_is_externally_visible_for_target(db, target))
}

fn parameter_owner_is_externally_visible_for_target(
    db: &dyn Db,
    resolved: &ResolvedDefinition,
) -> bool {
    let Some(definition) = resolved.definition() else {
        return false;
    };
    let parsed = parsed_module(db, definition.python_file(db));
    let target = definition.focus_range(db, &parsed.load(db));
    let module = parsed.load(db);

    let covering = covering_node(module.syntax().into(), target.range());
    let Ok(parameter_covering) =
        covering.find_last(|node| matches!(node, AnyNodeRef::Parameter(_)))
    else {
        return false;
    };

    let mut owner: Option<AnyNodeRef<'_>> = None;
    let mut seen_owner = false;
    let mut class_ancestor_found = false;

    // Heuristic: treat parameters as externally visible only when they belong to a top-level
    // function or a method on a top-level class. Nested functions/classes are excluded to avoid
    // broad, low-signal workspace scans.
    for ancestor in parameter_covering.ancestors() {
        if !seen_owner {
            if matches!(
                ancestor,
                AnyNodeRef::StmtFunctionDef(_) | AnyNodeRef::ExprLambda(_)
            ) {
                owner = Some(ancestor);
                seen_owner = true;
            }
            continue;
        }

        match ancestor {
            AnyNodeRef::StmtFunctionDef(_) | AnyNodeRef::ExprLambda(_) => {
                // Nested functions or lambdas are not externally visible.
                return false;
            }
            AnyNodeRef::StmtClassDef(_) => {
                if class_ancestor_found {
                    // Nested classes are treated as not externally visible for now.
                    return false;
                }
                class_ancestor_found = true;
            }
            _ => {}
        }
    }

    matches!(owner, Some(AnyNodeRef::StmtFunctionDef(_)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OccurrenceKind {
    /// An identifier that references a symbol.
    Reference,
    /// An identifier that declares a new symbol.
    Declaration,
    /// An identifier that binds a new value to a symbol.
    Binding,
}

impl OccurrenceKind {
    fn to_reference_kind(self) -> ReferenceKind {
        match self {
            Self::Reference => ReferenceKind::Read,
            Self::Declaration => ReferenceKind::Other,
            Self::Binding => ReferenceKind::Write,
        }
    }
}

impl From<ast::ExprContext> for OccurrenceKind {
    fn from(value: ast::ExprContext) -> Self {
        match value {
            ast::ExprContext::Load | ast::ExprContext::Invalid => Self::Reference,
            ast::ExprContext::Store | ast::ExprContext::Del => OccurrenceKind::Binding,
        }
    }
}

/// AST visitor that finds references by comparing each matching name's semantic definitions.
struct LocalReferencesFinder<'a> {
    model: &'a SemanticModel<'a>,
    tokens: &'a Tokens,
    searches: &'a LocalReferenceSearches<'a>,
    references: &'a mut Vec<ReferenceTarget>,
    mode: ReferencesMode,
    ancestors: Vec<AnyNodeRef<'a>>,
}

impl<'a> SourceOrderVisitor<'a> for LocalReferencesFinder<'a> {
    fn enter_node(&mut self, node: AnyNodeRef<'a>) -> TraversalSignal {
        self.ancestors.push(node);

        match node {
            AnyNodeRef::ExprName(name_expr) => {
                let target_text = name_expr.id.as_str();
                if self.searches.for_name(target_text).is_none() {
                    return TraversalSignal::Traverse;
                }

                let kind = OccurrenceKind::from(name_expr.ctx);
                let covering_node = CoveringNode::from_ancestors(self.ancestors.clone());
                self.check_covering_node(&covering_node, kind, target_text);
            }
            AnyNodeRef::ExprAttribute(attr_expr) => {
                let kind = OccurrenceKind::from(attr_expr.ctx);
                self.check_identifier(&attr_expr.attr, kind);
            }
            AnyNodeRef::StmtFunctionDef(func) => {
                self.check_declaration_identifier(&func.name);
            }
            AnyNodeRef::StmtClassDef(class) => {
                self.check_declaration_identifier(&class.name);
            }
            AnyNodeRef::Parameter(parameter) => {
                self.check_declaration_identifier(&parameter.name);
            }
            AnyNodeRef::Keyword(keyword) => {
                if let Some(arg) = &keyword.arg {
                    self.check_reference_identifier(arg);
                }
            }
            AnyNodeRef::StmtGlobal(global_stmt) => {
                for name in &global_stmt.names {
                    self.check_declaration_identifier(name);
                }
            }
            AnyNodeRef::StmtNonlocal(nonlocal_stmt) => {
                for name in &nonlocal_stmt.names {
                    self.check_declaration_identifier(name);
                }
            }
            AnyNodeRef::ExceptHandlerExceptHandler(handler) => {
                if let Some(name) = &handler.name {
                    self.check_binding_identifier(name);
                }
            }
            AnyNodeRef::PatternMatchAs(pattern_as) => {
                if let Some(name) = &pattern_as.name {
                    self.check_binding_identifier(name);
                }
            }
            AnyNodeRef::PatternMatchStar(pattern_star) => {
                if let Some(name) = &pattern_star.name {
                    self.check_binding_identifier(name);
                }
            }
            AnyNodeRef::PatternMatchMapping(pattern_mapping) => {
                if let Some(rest_name) = &pattern_mapping.rest {
                    self.check_binding_identifier(rest_name);
                }
            }
            AnyNodeRef::TypeParamParamSpec(param_spec) => {
                self.check_declaration_identifier(&param_spec.name);
            }
            AnyNodeRef::TypeParamTypeVarTuple(param_tuple) => {
                self.check_declaration_identifier(&param_tuple.name);
            }
            AnyNodeRef::TypeParamTypeVar(param_var) => {
                self.check_declaration_identifier(&param_var.name);
            }
            AnyNodeRef::ExprStringLiteral(string_expr) => {
                // A string literal listed in a class's `__slots__` names an
                // instance attribute, so renaming that attribute should rename
                // the matching slot string too.
                self.check_slots_string_literal(string_expr);

                // Highlight the sub-AST of a string annotation
                if let Some((sub_ast, sub_model)) = self.model.enter_string_annotation(string_expr)
                {
                    let mut sub_finder = LocalReferencesFinder {
                        model: &sub_model,
                        searches: self.searches,
                        references: self.references,
                        mode: self.mode,
                        tokens: sub_ast.tokens(),
                        ancestors: Vec::new(),
                    };
                    sub_finder.visit_expr(sub_ast.expr());
                }
            }
            AnyNodeRef::Alias(alias) => {
                // Handle import alias declarations
                if let Some(asname) = &alias.asname {
                    self.check_declaration_identifier(asname);
                }
                // Also check the original name for renames of the imported symbol itself.
                self.check_declaration_identifier(&alias.name);
            }
            _ => {}
        }

        TraversalSignal::Traverse
    }

    fn leave_node(&mut self, node: AnyNodeRef<'a>) {
        debug_assert_eq!(self.ancestors.last(), Some(&node));
        self.ancestors.pop();
    }
}

/// AST visitor that searches only keyword-argument labels for semantic matches against a target.
struct KeywordArgumentReferencesFinder<'a>(LocalReferencesFinder<'a>);

impl<'a> SourceOrderVisitor<'a> for KeywordArgumentReferencesFinder<'a> {
    fn enter_node(&mut self, node: AnyNodeRef<'a>) -> TraversalSignal {
        self.0.ancestors.push(node);

        if let AnyNodeRef::Keyword(keyword) = node {
            if let Some(arg) = &keyword.arg {
                self.0.check_reference_identifier(arg);
            }
        }

        TraversalSignal::Traverse
    }

    fn leave_node(&mut self, node: AnyNodeRef<'a>) {
        debug_assert_eq!(self.0.ancestors.last(), Some(&node));
        self.0.ancestors.pop();
    }
}

impl<'a> LocalReferencesFinder<'a> {
    /// Checks an identifier of a binding (e.g. `x = 10`)
    fn check_binding_identifier(&mut self, identifier: &ast::Identifier) {
        self.check_identifier(identifier, OccurrenceKind::Binding);
    }

    /// Checks an identifier that references a variable (a use).
    fn check_reference_identifier(&mut self, identifier: &ast::Identifier) {
        self.check_identifier(identifier, OccurrenceKind::Reference);
    }

    /// Checks an identifier that's part of a declaration, e.g. the name of the class.
    fn check_declaration_identifier(&mut self, identifier: &ast::Identifier) {
        self.check_identifier(identifier, OccurrenceKind::Declaration);
    }

    fn check_identifier(&mut self, identifier: &ast::Identifier, kind: OccurrenceKind) {
        let target_text = identifier.id.as_str();
        if self.searches.for_name(target_text).is_none() {
            return;
        }

        let mut ancestors_with_identifier = self.ancestors.clone();
        ancestors_with_identifier.push(AnyNodeRef::from(identifier));
        let covering_node = CoveringNode::from_ancestors(ancestors_with_identifier);
        self.check_covering_node(&covering_node, kind, target_text);
    }

    fn check_covering_node(
        &mut self,
        covering_node: &CoveringNode<'_>,
        kind: OccurrenceKind,
        target_text: &str,
    ) {
        let Some(targets) = self.searches.for_name(target_text) else {
            return;
        };

        // Use the start of the covering node as the offset. Any offset within
        // the node is fine here. Offsets matter only for import statements
        // where the identifier might be a multi-part module name.
        let offset = covering_node.node().start();
        let Some(goto_target) =
            GotoTarget::from_covering_node(self.model, covering_node, offset, self.tokens)
        else {
            return;
        };

        // One identifier name can participate in searches with both alias-resolution behaviors.
        // Resolve the occurrence once for each populated behavior and accept it if either
        // definition set intersects the corresponding targets.
        let matches = targets
            .iter()
            .any(|(import_alias_resolution, definitions)| {
                goto_target
                    .definitions(self.model, import_alias_resolution)
                    .and_then(|definitions| definitions.goto_declaration(self.model, &goto_target))
                    .is_some_and(|resolved| definitions.intersects(&resolved))
            });
        if !matches {
            return;
        }

        if matches!(self.mode, ReferencesMode::ReferencesSkipDeclaration) {
            let is_declaration = match kind {
                OccurrenceKind::Declaration => true,
                OccurrenceKind::Reference => false,
                OccurrenceKind::Binding => self.is_declaration(covering_node),
            };

            if is_declaration {
                return;
            }
        }

        let target = ReferenceTarget::new(
            self.model.file(),
            covering_node.node().range(),
            kind.to_reference_kind(),
        );
        self.references.push(target);
    }

    /// Checks a string literal that may be an entry in a class's `__slots__`.
    ///
    /// `__slots__` entries are plain strings, but they name instance
    /// attributes of the enclosing class. When the rename target is one of
    /// those attributes, the matching slot string should be renamed as well.
    /// We only do this when the attribute belongs to the same class that
    /// declares the `__slots__`, so unrelated classes that happen to use the
    /// same slot name are left untouched.
    fn check_slots_string_literal(&mut self, string_expr: &'a ast::ExprStringLiteral) {
        // Quick text-based check first. We only handle single-part string
        // literals; implicitly concatenated strings ("a" "b") can't name a
        // single attribute, so they never match an identifier here.
        if string_expr.value.is_implicit_concatenated() {
            return;
        }
        let [part] = string_expr.value.as_slice() else {
            return;
        };
        let target_text = part.value.as_ref();
        if self.searches.for_name(target_text).is_none() {
            return;
        }

        // The literal must sit directly inside a tuple/list/set container, or
        // be a key in a dict, that is the value of a `__slots__` assignment in
        // a class body.
        let Some(class) = self.enclosing_slots_class() else {
            return;
        };

        // Only rename the slot if the target attribute belongs to this class.
        if !self.target_belongs_to_class(class, target_text) {
            return;
        }

        // Rename the inner content of the string, leaving the quotes intact.
        let content_range = ast::StringLikePart::String(part).content_range();
        let target = ReferenceTarget::new(self.model.file(), content_range, ReferenceKind::Other);
        self.references.push(target);
    }

    /// Returns the class definition whose `__slots__` assignment contains the
    /// string literal currently being visited, if the ancestor chain matches
    /// one of the supported `__slots__` shapes.
    ///
    /// Supported shapes (v1):
    /// - `__slots__ = ("a", "b")` / `["a", "b"]` / `{"a", "b"}`
    /// - `__slots__ = {"a": ..., "b": ...}` (dict keys only)
    fn enclosing_slots_class(&self) -> Option<&'a ast::StmtClassDef> {
        // `self.ancestors` ends with the string literal itself. Walk outward.
        let mut ancestors = self.ancestors.iter().rev();

        // The string is either a direct element of a tuple/list/set, or a key
        // of a dict. Skip the immediate container node.
        match ancestors.next()? {
            AnyNodeRef::ExprStringLiteral(_) => {}
            _ => return None,
        }
        let container = *ancestors.next()?;
        match container {
            AnyNodeRef::ExprTuple(_) | AnyNodeRef::ExprList(_) | AnyNodeRef::ExprSet(_) => {}
            // For a dict, both keys and values have the `ExprDict` as their
            // direct parent, so `is_dict_key` checks that this string is one of
            // the keys before we treat it as a slot name.
            AnyNodeRef::ExprDict(dict) => {
                if !self.is_dict_key(dict) {
                    return None;
                }
            }
            _ => return None,
        }

        // The container must be the value of an assignment to `__slots__`.
        let assignment = ancestors.next()?;
        if !is_slots_assignment(*assignment, container) {
            return None;
        }

        // That assignment must live directly in a class body.
        match ancestors.next()? {
            AnyNodeRef::StmtClassDef(class) => Some(class),
            _ => None,
        }
    }

    /// Returns whether the string literal currently being visited is a *key*
    /// of `dict`, as opposed to one of its values.
    fn is_dict_key(&self, dict: &'a ast::ExprDict) -> bool {
        let Some(AnyNodeRef::ExprStringLiteral(string_expr)) = self.ancestors.last() else {
            return false;
        };
        dict.items.iter().any(|item| {
            item.key
                .as_ref()
                .is_some_and(|key| key.range() == string_expr.range())
        })
    }

    /// Returns whether any of the rename target's definitions is an instance attribute of `class`.
    ///
    /// The target must name an actual instance attribute of `class`, either a member access like
    /// `self.value = ...` or a class-body declaration like `value: int` (optionally using `...` as
    /// the value in a stub), and its nearest enclosing class must be `class` itself. A parameter or
    /// local that merely shares the name, or an attribute of a nested class, is not treated as the
    /// slot.
    fn target_belongs_to_class(&self, class: &'a ast::StmtClassDef, target_text: &str) -> bool {
        let db = self.model.db();
        let file = self.model.file();
        let class_range = class.range();
        let module = ruff_db::parsed::parsed_module(db, self.model.python_file()).load(db);
        let index = ty_python_core::semantic_index(db, self.model.program_file());

        // The nearest class scope lexically enclosing `scope`, if any. `ancestor_scopes` skips
        // class scopes for name resolution, so we walk the lexical parents directly to stop at the
        // innermost enclosing class rather than an outer one.
        let nearest_enclosing_class_scope = |mut scope: FileScopeId| loop {
            let node = index.scope(scope);
            if node.kind() == ScopeKind::Class {
                return Some(scope);
            }
            scope = node.parent()?;
        };

        let Some(targets) = self.searches.for_name(target_text) else {
            return false;
        };

        targets.definitions().any(|resolved| {
            let Some(definition) = resolved.definition() else {
                return false;
            };
            if definition.file(db) != file {
                return false;
            }

            // The target's nearest enclosing class must be the one declaring the `__slots__`, so an
            // attribute of a nested class doesn't rename an outer class's slot.
            let definition_scope = definition.file_scope(db);
            let Some(owning_class_scope) = nearest_enclosing_class_scope(definition_scope) else {
                return false;
            };
            match index.scope(owning_class_scope).node() {
                NodeWithScopeKind::Class(node) if node.node(&module).range() == class_range => {}
                _ => return false,
            }

            // Accept only a member access (`self.value = ...`) or a class-body attribute declaration
            // (`value: int` or `value: int = ...` in a stub), so a parameter or local sharing the
            // name is not treated as the slot.
            let place = definition.place(db);
            let is_class_attribute_declaration = definition_scope == owning_class_scope
                && place.is_symbol()
                && matches!(
                    definition.kind(db),
                    DefinitionKind::AnnotatedAssignment(assignment)
                        if assignment.value(&module).is_none_or(|value| {
                            file.is_stub(db) && value.is_ellipsis_literal_expr()
                        })
                );
            place.is_member() || is_class_attribute_declaration
        })
    }

    fn is_declaration(&self, covering_node: &CoveringNode<'_>) -> bool {
        let db = self.model.db();

        let Some(local_definition) = self.model.first_local_definition(covering_node) else {
            return false;
        };

        let file = local_definition.file(db);
        let module = ruff_db::parsed::parsed_module(db, local_definition.python_file(db)).load(db);
        let kind = local_definition.kind(db);
        let category = kind.category(file.is_stub(db), &module);

        if category.is_declaration() {
            return true;
        }

        if self.binding_has_reachable_explicit_declaration(local_definition) {
            return false;
        }

        self.binding_is_first_assignment_on_some_path(local_definition)
    }

    fn binding_has_reachable_explicit_declaration(&self, binding: Definition<'a>) -> bool {
        let db = self.model.db();
        let use_def = ty_python_core::use_def_map(db, binding.scope(db));
        use_def
            .declarations_at_binding(binding)
            .any(|declaration| declaration.declaration.definition().is_some())
    }

    fn binding_is_first_assignment_on_some_path(&self, binding: Definition<'a>) -> bool {
        let db = self.model.db();
        let use_def = ty_python_core::use_def_map(db, binding.scope(db));
        use_def
            .bindings_at_definition(binding)
            .any(|prior_binding| {
                matches!(
                    prior_binding.binding,
                    DefinitionState::Deleted | DefinitionState::Undefined
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goto::find_goto_target;
    use crate::tests::{CursorTest, cursor_test};

    fn cursor_target_is_externally_visible(test: &CursorTest) -> bool {
        let model = SemanticModel::new(&test.db, test.program_file(test.cursor.file));
        let goto_target =
            find_goto_target(&model, &test.cursor.parsed, test.cursor.offset).unwrap();
        let definitions = goto_target
            .definitions(
                &model,
                ReferencesMode::References.to_import_alias_resolution(),
            )
            .unwrap();

        has_any_external_visible_definitions(&test.db, &definitions)
    }

    #[test]
    fn externally_visible_definitions_can_have_cross_file_references() {
        for (case, source) in [
            ("module-global", "x<CURSOR> = 1"),
            (
                "module comprehension walrus",
                "[(x<CURSOR> := item) for item in [1]]",
            ),
            (
                "nested module comprehension walrus",
                "[[(x<CURSOR> := item) for item in [1]] for _ in [1]]",
            ),
            (
                "explicit global comprehension walrus",
                "
x = 0
def f():
    global x
    [(x<CURSOR> := item) for item in [1]]
",
            ),
            (
                "class",
                "
class C:
    x<CURSOR> = 1
",
            ),
        ] {
            let test = cursor_test(source);
            assert!(cursor_target_is_externally_visible(&test), "{case}");
        }
    }

    #[test]
    fn non_public_scope_definitions_stay_in_file() {
        for (case, source) in [
            (
                "function",
                "
def f():
    x<CURSOR> = 1
    return x
",
            ),
            ("lambda", "f = lambda x<CURSOR>: x"),
            ("comprehension", "xs = [x for x<CURSOR> in range(3)]"),
            (
                "function comprehension walrus",
                "
def f():
    [(x<CURSOR> := item) for item in [1]]
    return x
",
            ),
            (
                "nested function comprehension walrus",
                "
def f():
    [[(x<CURSOR> := item) for item in [1]] for _ in [1]]
    return x
",
            ),
            (
                "lambda comprehension walrus",
                "f = lambda: [(x<CURSOR> := item) for item in [1]]",
            ),
            (
                "explicit nonlocal comprehension walrus",
                "
def outer():
    x = 0
    def inner():
        nonlocal x
        [(x<CURSOR> := item) for item in [1]]
    inner()
    return x
",
            ),
            ("type parameters", "type Alias[T<CURSOR>] = list[T]"),
        ] {
            let test = cursor_test(source);
            assert!(!cursor_target_is_externally_visible(&test), "{case}");
        }
    }
}
