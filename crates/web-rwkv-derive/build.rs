use cargo_metadata::MetadataCommand;
use std::{
    collections::{HashSet, VecDeque},
    path::Path,
};

fn main() {
    let metadata = MetadataCommand::new()
        .exec()
        .expect("failed to obtain cargo metadata");

    // locate the current package using CARGO_MANIFEST_DIR
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let manifest_path = Path::new(&manifest_dir).join("Cargo.toml");
    let manifest_path_str = manifest_path.to_str().expect("path not UTF-8");

    let current_package = metadata
        .packages
        .iter()
        .find(|p| p.manifest_path.as_str() == manifest_path_str)
        .expect("current package not found in metadata");

    let resolve = metadata.resolve.as_ref().expect("resolve graph missing");

    let current_node = resolve
        .nodes
        .iter()
        .find(|node| node.id == current_package.id)
        .expect("current node not found in resolve graph");

    // perform BFS to find the first occurrence of the "serde" crate
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    queue.push_back(current_node);
    visited.insert(&current_node.id);

    let serde_version = loop {
        let node = queue.pop_front().expect("dependency graph exhausted");

        // check dependencies of the current node
        let mut found = None;
        for id in &node.dependencies {
            if visited.contains(id) {
                continue;
            }
            visited.insert(id);

            let node = resolve
                .nodes
                .iter()
                .find(|n| &n.id == id)
                .expect("dependency node not found");
            let package = metadata
                .packages
                .iter()
                .find(|p| p.id == node.id)
                .expect("dependency package not found");

            if package.name == "serde" {
                found = Some(package.version.clone());
                break;
            }
            queue.push_back(node);
        }

        if let Some(version) = found {
            break version;
        }
    };

    println!(
        "cargo:rustc-env=SERDE_PATCH_VERSION={}",
        serde_version.patch
    );
}
