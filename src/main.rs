use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use clap::Parser;
use colored::Colorize;
use itertools::Itertools;
use log::*;
use protobuf::Message;
use scip::types::Occurrence;
use scip::types::{symbol_information::Kind, Document, SymbolInformation, SymbolRole};

/// Check whether the function/method defined at `def_line` (0-indexed) has a
/// `#[test]`, `#[main]`, `#[tokio::main]`, or similar attribute.
/// Also returns true for plain `fn main()` (entry point, never explicitly called).
fn has_test_or_main_attr(lines: &[impl AsRef<str>], def_line: usize, display_name: &str) -> bool {
    // Plain main is always filtered (entry point)
    if display_name == "main" {
        return true;
    }
    // Scan upward from the line before the definition, up to 10 lines
    let start = def_line.saturating_sub(10);
    for i in (start..def_line).rev() {
        let trimmed = lines[i].as_ref().trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if trimmed.starts_with("#[") {
            // Match test/main attributes but avoid false positives from
            // unrelated attributes that happen to contain "test"/"main" as
            // substrings (e.g. #[serde(rename = "contest")]).
            // We check for common patterns: #[test], #[..::test], #[..::main]
            let attr_body = trimmed.trim_start_matches("#[").trim_end_matches(']');
            let attr_name = attr_body.split('(').next().unwrap_or(attr_body).trim();
            // attr_name is e.g. "test", "tokio::test", "tokio::main", "actix_web::main"
            if attr_name == "test"
                || attr_name.ends_with("::test")
                || attr_name == "main"
                || attr_name.ends_with("::main")
            {
                return true;
            }
            // Other attributes — keep scanning
            continue;
        }
        // Lines that can appear between attributes and the fn definition
        if trimmed.starts_with("pub")
            || trimmed.starts_with("async")
            || trimmed.starts_with("unsafe")
            || trimmed.starts_with("fn ")
            || trimmed.starts_with("const ")
        {
            continue;
        }
        // Hit something else (e.g. previous function body) — stop
        break;
    }
    false
}

/// Check whether the given 0-indexed `line` falls inside a `#[cfg(test)]` module.
/// Uses simple brace-counting from the top of the file.
fn is_inside_cfg_test(lines: &[impl AsRef<str>], line: usize) -> bool {
    let mut cfg_test_depth: usize = 0;
    let mut in_cfg_test = false;

    for (i, l) in lines.iter().enumerate() {
        if i > line {
            break;
        }
        let trimmed = l.as_ref().trim();

        if trimmed.starts_with("#[cfg(test)]") {
            in_cfg_test = true;
            cfg_test_depth = 0;
        }

        if in_cfg_test {
            // Record the flag before processing braces (closing brace is still inside)
            let was_in_cfg_test = true;
            for ch in l.as_ref().chars() {
                if ch == '{' {
                    cfg_test_depth += 1;
                } else if ch == '}' {
                    if cfg_test_depth <= 1 {
                        in_cfg_test = false;
                        cfg_test_depth = 0;
                    } else {
                        cfg_test_depth -= 1;
                    }
                }
            }
            if i == line {
                return was_in_cfg_test;
            }
        } else if i == line {
            return false;
        }
    }
    false
}

/// For each line in the file, compute whether it falls inside a `#[cfg(test)]` block.
/// Returns a Vec<bool> of the same length as `lines`.
#[cfg(test)]
fn compute_cfg_test_flags(lines: &[impl AsRef<str>]) -> Vec<bool> {
    let mut flags = vec![false; lines.len()];
    let mut cfg_test_depth: usize = 0;
    let mut in_cfg_test = false;

    for (i, l) in lines.iter().enumerate() {
        let trimmed = l.as_ref().trim();

        if trimmed.starts_with("#[cfg(test)]") {
            in_cfg_test = true;
            cfg_test_depth = 0;
        }

        if in_cfg_test {
            flags[i] = true;
            for ch in l.as_ref().chars() {
                if ch == '{' {
                    cfg_test_depth += 1;
                } else if ch == '}' {
                    if cfg_test_depth <= 1 {
                        in_cfg_test = false;
                        cfg_test_depth = 0;
                    } else {
                        cfg_test_depth -= 1;
                    }
                }
            }
        }
    }
    flags
}

/// Check whether a file path (relative) looks like it belongs to test code.
/// Matches paths under `tests/` directories.
fn is_test_file(relative_path: &str) -> bool {
    let path = std::path::Path::new(relative_path);
    path.components().any(|c| c.as_os_str() == "tests")
}

/// Check whether a SCIP symbol represents a trait impl method.
/// rust-analyzer encodes trait impl methods as `impl#[StructName][TraitName]method()`,
/// with two `[...]` brackets after `impl#`, while inherent impl methods have only one:
/// `impl#[StructName]method()`.
fn is_trait_impl_method(symbol: &str) -> bool {
    if let Some(after_impl) = symbol.rsplit("impl#").next() {
        // Count the number of `[...]` segments before the method descriptor
        let bracket_count = after_impl.matches('[').count();
        bracket_count >= 2
    } else {
        false
    }
}

#[derive(Parser)]
#[command(name = "cargo")]
#[command(bin_name = "cargo")]
enum MainFlags {
    WorkspaceUnusedPub(Flags),
}

/// Detect unused pub methods in a workspace.
#[derive(clap::Args)]
#[command(version, about)]
struct Flags {
    #[clap(default_value_os_t = std::env::current_dir().unwrap())]
    workspace: PathBuf,
    #[clap(long)]
    scip: Option<PathBuf>,
    /// Treat pub items that are only referenced from test code as unused.
    /// Test code includes `#[cfg(test)]` modules and files under `tests/` directories.
    #[clap(long)]
    skip_test_usages: bool,
    /// Force regeneration of the SCIP index even if it already exists.
    #[clap(long)]
    force_regenerate: bool,
    /// Suppress log output (info, warn, debug messages).
    #[clap(long, short)]
    quiet: bool,
}

fn main_impl(args: MainFlags) -> anyhow::Result<()> {
    let MainFlags::WorkspaceUnusedPub(args) = args;
    let default_level = if args.quiet { "error" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_level))
        .init();

    let scip = args
        .scip
        .unwrap_or_else(|| args.workspace.join("index.scip"));

    if !args.workspace.join("Cargo.toml").exists() {
        anyhow::bail!("{:?} does not contain a Cargo.toml file", args.workspace);
    }
    if args.force_regenerate && scip.exists() {
        warn!(
            "Force regeneration requested. Removing existing SCIP file at {:?}",
            scip
        );
        std::fs::remove_file(&scip)?;
    }
    if !scip.exists() {
        warn!(
            "SCIP file not found at {:?}. Generating with rust-analyzer. This may take a while for large workspaces.",
            scip
        );
        duct::cmd!("rust-analyzer", "scip", &args.workspace, "--output", &scip)
            .dir(&args.workspace)
            .stdout_null()
            .stderr_null()
            .run()?;
    }
    info!("Running on {:?} with SCIP {:?}", args.workspace, scip);

    // Parse SCIP
    let reader = std::fs::File::open(scip)?;
    let mut reader = std::io::BufReader::new(reader);
    let index = scip::types::Index::parse_from_reader(&mut reader)?;
    debug!("Opened SCIP file with {} documents", index.documents.len());

    // Record declarations (keyed by SCIP symbol, not display_name)
    let mut declarations = HashMap::<&String, &SymbolInformation>::default();
    for doc in &index.documents {
        for s in &doc.symbols {
            let Ok(kind) = s.kind.enum_value() else {
                continue;
            };
            if !matches!(
                kind,
                Kind::Method
                    | Kind::Function
                    | Kind::StaticMethod
                    | Kind::Constant
                    | Kind::Struct
                    | Kind::Enum
                    | Kind::EnumMember
                    | Kind::Trait
                    | Kind::Type
                    | Kind::TypeAlias
                    | Kind::Field
                    | Kind::Macro
                    | Kind::Union
                    | Kind::StaticVariable
            ) {
                continue;
            }
            // Skip trait impl methods — their pub visibility is dictated by the
            // trait, not by the developer, so they are not truly "unused pub".
            if is_trait_impl_method(&s.symbol) {
                continue;
            }
            declarations.insert(&s.symbol, s);
        }
    }
    debug!("Found {} declarations", declarations.len());

    // Record occurrences
    if args.skip_test_usages {
        // Track which symbols have non-test references
        let mut has_non_test_ref: HashSet<&String> = HashSet::new();
        for doc in &index.documents {
            let is_test_path = is_test_file(&doc.relative_path);
            // Lazily load file lines for cfg(test) detection
            let mut doc_lines: Option<Vec<String>> = None;
            for o in &doc.occurrences {
                if (o.symbol_roles & SymbolRole::Definition as i32) == 0
                    && declarations.contains_key(&o.symbol)
                {
                    let in_test = if is_test_path {
                        true
                    } else {
                        let lines = doc_lines.get_or_insert_with(|| {
                            let full = args.workspace.join(&doc.relative_path);
                            std::fs::read_to_string(&full)
                                .unwrap_or_default()
                                .lines()
                                .map(String::from)
                                .collect()
                        });
                        is_inside_cfg_test(lines, o.range[0] as usize)
                    };
                    if !in_test {
                        has_non_test_ref.insert(&o.symbol);
                    }
                }
            }
        }
        // Remove only symbols that have non-test references
        declarations.retain(|k, _| !has_non_test_ref.contains(k));
    } else {
        for doc in &index.documents {
            for o in &doc.occurrences {
                if (o.symbol_roles & SymbolRole::Definition as i32) == 0 {
                    declarations.remove(&o.symbol);
                }
            }
        }
    }

    debug!("Pass 1: {} candidates", declarations.len());

    // Collect definition occurrences for Pass-1 candidates so we can inspect
    // source attributes in Pass 2.
    let mut def_positions: Vec<(&Document, &Occurrence, &SymbolInformation)> = vec![];
    for doc in &index.documents {
        for o in &doc.occurrences {
            if let Some(info) = declarations.get(&o.symbol) {
                if (o.symbol_roles & SymbolRole::Definition as i32) > 0 {
                    def_positions.push((doc, o, info));
                }
            }
        }
    }

    // Cache source file contents keyed by relative path
    let mut file_cache: HashMap<&str, Vec<String>> = HashMap::new();

    // Pass 2
    // Remove:
    //   - functions with #[test], #[main], #[tokio::main] attributes (or fn main)
    //   - trait methods (which may be called implicitly)
    let mut symbols_to_remove: HashSet<&String> = HashSet::new();
    for (doc, occ, info) in &def_positions {
        let rel_path: &str = &doc.relative_path;
        let full_path = args.workspace.join(rel_path);
        let lines = file_cache.entry(rel_path).or_insert_with(|| {
            std::fs::read_to_string(&full_path)
                .unwrap_or_default()
                .lines()
                .map(String::from)
                .collect()
        });
        let def_line = occ.range[0] as usize;

        // Skip items defined inside test code when --skip-test-usages is set
        if args.skip_test_usages
            && (is_test_file(rel_path)
                || (def_line < lines.len() && is_inside_cfg_test(lines, def_line)))
        {
            symbols_to_remove.insert(&info.symbol);
            continue;
        }

        // Attribute-based test/main check
        if def_line < lines.len() && has_test_or_main_attr(lines, def_line, &info.display_name) {
            symbols_to_remove.insert(&info.symbol);
            continue;
        }

        // For struct fields, only report pub fields
        if info.kind.enum_value() == Ok(Kind::Field) && def_line < lines.len() {
            let trimmed = lines[def_line].trim();
            if !trimmed.starts_with("pub ") && !trimmed.starts_with("pub(") {
                symbols_to_remove.insert(&info.symbol);
            }
        }
    }
    declarations.retain(|k, _| !symbols_to_remove.contains(k));
    debug!(
        "Pass 2 (mains, tests, trait methods): {} candidates",
        declarations.len()
    );

    let n_found = declarations.len();
    info!("Found {} possibly unused pub items", n_found);

    // Reuse definition positions collected earlier, filtered to remaining candidates
    let declarations_occurrences: Vec<(&String, &Occurrence)> = def_positions
        .iter()
        .filter(|(_, _, info)| declarations.contains_key(&info.symbol))
        .map(|(doc, occ, _)| (&doc.relative_path, *occ))
        .collect();
    // Group by file
    let mut declarations_occurrences = declarations_occurrences
        .into_iter()
        .into_group_map()
        .into_iter()
        .collect_vec();
    declarations_occurrences.sort_by_key(|(d, _)| *d);
    // Display
    for (path, mut occs) in declarations_occurrences {
        let full_path = args.workspace.join(path);
        if !full_path.exists() {
            warn!("{} not found, is the SCIP file up-to-date?", path);
            continue;
        }
        let lines = std::fs::read_to_string(full_path)?;
        let lines: Vec<&str> = lines.lines().collect();
        occs.sort_by_key(|occ| occ.range[0]);
        println!("{}", path.yellow());
        for occ in occs {
            let line = occ.range[0] as usize;
            println!("{:<4} {}", (line + 1).to_string().blue(), lines[line]);
        }
        println!();
    }
    anyhow::ensure!(n_found == 0, "Found {} possibly unused pub items", n_found);
    Ok(())
}

fn main() {
    if let Err(e) = main_impl(MainFlags::parse()) {
        error!("{}", e);
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── has_test_or_main_attr ──────────────────────────────────────────

    #[test]
    fn detects_test_attribute() {
        let lines = vec!["    #[test]", "    fn test_foo() {"];
        assert!(has_test_or_main_attr(&lines, 1, "test_foo"));
    }

    #[test]
    fn detects_tokio_test_attribute() {
        let lines = vec!["    #[tokio::test]", "    async fn it_works() {"];
        assert!(has_test_or_main_attr(&lines, 1, "it_works"));
    }

    #[test]
    fn detects_main_by_name() {
        let lines = vec!["fn main() {"];
        assert!(has_test_or_main_attr(&lines, 0, "main"));
    }

    #[test]
    fn detects_tokio_main_attribute() {
        let lines = vec!["#[tokio::main]", "async fn main() {"];
        assert!(has_test_or_main_attr(&lines, 1, "main"));
    }

    #[test]
    fn detects_actix_main_attribute() {
        let lines = vec!["#[actix_web::main]", "async fn main() {"];
        assert!(has_test_or_main_attr(&lines, 1, "main"));
    }

    #[test]
    fn detects_test_with_doc_comments() {
        let lines = vec!["/// This is a doc comment", "#[test]", "fn my_test() {"];
        assert!(has_test_or_main_attr(&lines, 2, "my_test"));
    }

    #[test]
    fn detects_test_with_multiple_attributes() {
        let lines = vec!["#[allow(unused)]", "#[test]", "fn my_test() {"];
        assert!(has_test_or_main_attr(&lines, 2, "my_test"));
    }

    #[test]
    fn no_false_positive_for_regular_function() {
        let lines = vec!["}", "", "pub fn do_something() {"];
        assert!(!has_test_or_main_attr(&lines, 2, "do_something"));
    }

    #[test]
    fn no_false_positive_when_test_is_in_another_function() {
        let lines = vec![
            "    #[test]",
            "    fn test_other() {",
            "        assert!(true);",
            "    }",
            "",
            "    pub fn helper() {",
        ];
        // helper is at line 5; #[test] at line 0 belongs to test_other,
        // scanning stops at "}" on line 3
        assert!(!has_test_or_main_attr(&lines, 5, "helper"));
    }

    #[test]
    fn detects_test_on_pub_async_fn() {
        let lines = vec!["#[test]", "pub async fn test_async() {"];
        assert!(has_test_or_main_attr(&lines, 1, "test_async"));
    }

    #[test]
    fn no_false_positive_for_constant() {
        let lines = vec!["pub const MAX_RETRIES: u32 = 3;"];
        assert!(!has_test_or_main_attr(&lines, 0, "MAX_RETRIES"));
    }

    #[test]
    fn def_line_zero_no_scan() {
        // When def_line is 0, there are no lines above to scan
        let lines = vec!["fn regular() {"];
        assert!(!has_test_or_main_attr(&lines, 0, "regular"));
    }

    // ── Kind acceptance ─────────────────────────────────────────────

    fn kind_accepted(kind: Kind) -> bool {
        matches!(
            kind,
            Kind::Method
                | Kind::Function
                | Kind::StaticMethod
                | Kind::Constant
                | Kind::Struct
                | Kind::Enum
                | Kind::EnumMember
                | Kind::Trait
                | Kind::Type
                | Kind::TypeAlias
                | Kind::Field
                | Kind::Macro
                | Kind::Union
                | Kind::StaticVariable
        )
    }

    #[test]
    fn accepted_kinds() {
        let accepted = [
            Kind::Method,
            Kind::Function,
            Kind::StaticMethod,
            Kind::Constant,
            Kind::Struct,
            Kind::Enum,
            Kind::EnumMember,
            Kind::Trait,
            Kind::Type,
            Kind::TypeAlias,
            Kind::Field,
            Kind::Macro,
            Kind::Union,
            Kind::StaticVariable,
        ];
        for kind in &accepted {
            assert!(kind_accepted(*kind), "Kind {:?} should be accepted", kind);
        }
    }

    #[test]
    fn rejected_kinds() {
        let rejected = [Kind::Variable, Kind::Package, Kind::Module];
        for kind in &rejected {
            assert!(!kind_accepted(*kind), "Kind {:?} should be rejected", kind);
        }
    }

    // ── SCIP integration-style tests ──────────────────────────────────

    fn make_symbol_info(symbol: &str, display_name: &str, kind: Kind) -> SymbolInformation {
        let mut info = SymbolInformation::new();
        info.symbol = symbol.to_string();
        info.display_name = display_name.to_string();
        info.kind = protobuf::EnumOrUnknown::new(kind);
        info
    }

    fn make_occurrence(symbol: &str, line: i32, roles: i32) -> Occurrence {
        let mut occ = Occurrence::new();
        occ.symbol = symbol.to_string();
        occ.range = vec![line, 0, line, 10];
        occ.symbol_roles = roles;
        occ
    }

    #[test]
    fn pass1_removes_referenced_symbols() {
        // A symbol with a non-definition occurrence should be removed
        let sym = "rust-analyzer crate 0.1.0 Foo#bar().";
        let info = make_symbol_info(sym, "bar", Kind::Method);
        let mut declarations = HashMap::new();
        declarations.insert(&info.symbol, &info);

        // Simulate an occurrence that is NOT a definition (reference)
        let occ = make_occurrence(sym, 10, 0);
        // Pass 1 logic: remove if occurrence is not a definition
        if (occ.symbol_roles & SymbolRole::Definition as i32) == 0 {
            declarations.remove(&occ.symbol);
        }
        assert!(
            declarations.is_empty(),
            "Referenced symbol should be removed"
        );
    }

    #[test]
    fn pass1_keeps_unreferenced_symbols() {
        let sym = "rust-analyzer crate 0.1.0 Foo#unused_bar().";
        let info = make_symbol_info(sym, "unused_bar", Kind::Method);
        let mut declarations = HashMap::new();
        declarations.insert(&info.symbol, &info);

        // Only a definition occurrence — should NOT be removed
        let occ = make_occurrence(sym, 5, SymbolRole::Definition as i32);
        if (occ.symbol_roles & SymbolRole::Definition as i32) == 0 {
            declarations.remove(&occ.symbol);
        }
        assert_eq!(declarations.len(), 1, "Unreferenced symbol should remain");
    }

    // ── is_trait_impl_method ────────────────────────────────────────

    #[test]
    fn detects_trait_impl_method_two_brackets() {
        // impl#[StructName][TraitName]method() — trait impl
        assert!(is_trait_impl_method(
            "rust-analyzer cargo crate 0.1.0 impl#[SqliteTraceStore][TraceStore]save_trace()."
        ));
    }

    #[test]
    fn detects_external_trait_impl_method() {
        // External trait (e.g. Display)
        assert!(is_trait_impl_method(
            "rust-analyzer cargo crate 0.1.0 impl#[MyStruct][`std::fmt::Display`]fmt()."
        ));
    }

    #[test]
    fn inherent_impl_method_not_filtered() {
        // impl#[StructName]method() — inherent impl, only one bracket pair
        assert!(!is_trait_impl_method(
            "rust-analyzer cargo crate 0.1.0 impl#[SqlitePool]open()."
        ));
    }

    #[test]
    fn non_impl_symbol_not_filtered() {
        // Regular function, no impl# at all
        assert!(!is_trait_impl_method(
            "rust-analyzer cargo crate 0.1.0 my_function()."
        ));
    }

    #[test]
    fn all_supported_kinds_collected_as_declarations() {
        for kind in [
            Kind::Method,
            Kind::Function,
            Kind::StaticMethod,
            Kind::Constant,
            Kind::Struct,
            Kind::Enum,
            Kind::EnumMember,
            Kind::Trait,
            Kind::Type,
            Kind::TypeAlias,
            Kind::Field,
            Kind::Macro,
            Kind::Union,
            Kind::StaticVariable,
        ] {
            assert!(kind_accepted(kind), "Kind {:?} should be collected", kind);
        }
    }

    // ── is_inside_cfg_test / compute_cfg_test_flags ──────────────────

    #[test]
    fn cfg_test_detects_line_inside_test_module() {
        let lines = vec![
            "fn production() {}",
            "#[cfg(test)]",
            "mod tests {",
            "    fn test_helper() {}",
            "}",
            "fn after() {}",
        ];
        assert!(!is_inside_cfg_test(&lines, 0));
        assert!(is_inside_cfg_test(&lines, 1));
        assert!(is_inside_cfg_test(&lines, 2));
        assert!(is_inside_cfg_test(&lines, 3));
        assert!(is_inside_cfg_test(&lines, 4)); // closing `}` is still part of the module
        assert!(!is_inside_cfg_test(&lines, 5));
    }

    #[test]
    fn cfg_test_handles_nested_braces() {
        let lines = vec![
            "#[cfg(test)]",
            "mod tests {",
            "    fn helper() {",
            "        let x = 1;",
            "    }",
            "    fn another() {}",
            "}",
            "fn outside() {}",
        ];
        assert!(is_inside_cfg_test(&lines, 3));
        assert!(is_inside_cfg_test(&lines, 5));
        assert!(!is_inside_cfg_test(&lines, 7));
    }

    #[test]
    fn compute_cfg_test_flags_matches_is_inside() {
        let lines = vec![
            "fn prod() {}",
            "#[cfg(test)]",
            "mod tests {",
            "    fn t() {}",
            "}",
            "fn after() {}",
        ];
        let flags = compute_cfg_test_flags(&lines);
        assert_eq!(flags, vec![false, true, true, true, true, false]);
    }

    // ── is_test_file ─────────────────────────────────────────────────

    #[test]
    fn test_file_detection() {
        assert!(is_test_file("tests/integration.rs"));
        assert!(is_test_file("crate_a/tests/foo.rs"));
        assert!(!is_test_file("src/main.rs"));
        assert!(!is_test_file("src/tests_helper.rs"));
    }
}
