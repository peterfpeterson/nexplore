use crate::widgets::tree::TreeItem;
use anyhow::{anyhow, Context};
use hdf5::{
    dataset::Layout, filters::Filter, types::TypeDescriptor, Dataset, File, Group, LinkInfo,
    LinkType, Location,
};
use std::collections::HashMap;
use std::{fmt::Display, path::Path};

#[cfg(test)]
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum EntityInfo {
    Group(GroupInfo),
    Dataset(DatasetInfo),
}

impl From<EntityInfo> for TreeItem<'_> {
    fn from(value: EntityInfo) -> Self {
        match value {
            EntityInfo::Group(info) => TreeItem::from(info),
            EntityInfo::Dataset(info) => TreeItem::from(info),
        }
    }
}

pub fn get_attrs(location: &Location) -> HashMap<String, String> {
    let mut attrs = HashMap::new();
    if let Ok(attr_names) = location.attr_names() {
        for name in attr_names {
            let value = location.attr(&name).unwrap();
            attrs.insert(name, format!("{:?}", value));
        }
    };
    attrs
}

#[derive(Debug, Clone)]
pub struct GroupInfo {
    pub name: String,
    pub id: i64,
    pub link_kind: LinkKind,
    pub entities: Vec<EntityInfo>,
    pub attrs: HashMap<String, String>,
}

impl GroupInfo {
    fn try_from_group_and_link(group: Group, link: LinkInfo) -> Result<Self, anyhow::Error> {
        let name = group.name().split('/').next_back().unwrap().to_string();
        let id = group.id();
        let attrs = get_attrs(&group);
        let entities = group
            .iter_visit_default(Vec::new(), |group, key, link, entities| {
                let entity = if let Ok(group) = group.group(key) {
                    GroupInfo::try_from_group_and_link(group, link).map(EntityInfo::Group)
                } else if let Ok(dataset) = group.dataset(key) {
                    Ok(EntityInfo::Dataset(DatasetInfo::from_dataset_and_link(
                        dataset, link,
                    )))
                } else {
                    Err(anyhow!("Found link to entity of unknown kind"))
                };
                entities.push(entity);
                true
            })?
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            name,
            id,
            link_kind: link.link_type.into(),
            entities,
            attrs,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub name: String,
    pub id: i64,
    pub link_type: LinkKind,
    pub shape: Vec<usize>,
    pub layout_info: DatasetLayoutInfo,
    pub dtype_descr: TypeDescriptor,
    pub attrs: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum DatasetLayoutInfo {
    Compact {},
    Contiguous {},
    Chunked {
        chunk_shape: Vec<usize>,
        filters: Vec<Filter>,
    },
    Virtial {},
}

impl DatasetInfo {
    fn from_dataset_and_link(dataset: Dataset, link: LinkInfo) -> Self {
        let name = dataset.name().split('/').next_back().unwrap().to_string();
        let id = dataset.id();
        let shape = dataset.shape();
        let layout_info = match dataset.layout() {
            Layout::Compact => DatasetLayoutInfo::Compact {},
            Layout::Contiguous => DatasetLayoutInfo::Contiguous {},
            Layout::Chunked => DatasetLayoutInfo::Chunked {
                chunk_shape: dataset.chunk().unwrap(),
                filters: dataset.filters(),
            },
            Layout::Virtual => DatasetLayoutInfo::Virtial {},
        };
        let dtype_descr = dataset.dtype().unwrap().to_descriptor().unwrap();
        let attrs = get_attrs(&dataset);

        Self {
            name,
            id,
            link_type: link.link_type.into(),
            shape,
            layout_info,
            dtype_descr,
            attrs,
        }
    }
}

#[derive(Debug, Clone)]
pub enum LinkKind {
    Hard,
    Soft,
    External,
}

impl From<LinkType> for LinkKind {
    fn from(value: LinkType) -> Self {
        match value {
            LinkType::Hard => Self::Hard,
            LinkType::Soft => Self::Soft,
            LinkType::External => Self::External,
        }
    }
}

impl Display for LinkKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Hard => "Hard",
            Self::Soft => "Soft",
            Self::External => "External",
        })
    }
}

#[derive(Debug, Clone)]
pub struct FileInfo {
    pub name: String,
    pub size: u64,
    pub entities: Vec<EntityInfo>,
}

impl FileInfo {
    pub fn read(path: impl AsRef<Path>) -> Result<Self, anyhow::Error> {
        let name = path
            .as_ref()
            .file_name()
            .context("No file in path")?
            .to_string_lossy()
            .into_owned();
        let file = File::open(path)?;
        let size = file.size();
        let entities = GroupInfo::try_from_group_and_link(
            file.as_group()?,
            LinkInfo {
                link_type: LinkType::Hard,
                creation_order: None,
                is_utf8: true,
            },
        )?
        .entities;

        Ok(Self {
            name,
            size,
            entities,
        })
    }

    pub fn entity(&self, index: Vec<usize>) -> Result<EntityInfo, anyhow::Error> {
        let mut indices = index.into_iter();
        let mut entity = self
            .entities
            .get(indices.next().context("Index was empty")?)
            .context("No entity at index")?;
        for idx in indices {
            match entity {
                EntityInfo::Group(group) => {
                    entity = group.entities.get(idx).context("Index was empty")?
                }
                EntityInfo::Dataset(_) => Err(anyhow!("Cannot index into a dataset"))?,
            }
        }
        Ok(entity.clone())
    }

    pub fn to_tree_items(&self) -> Vec<TreeItem<'_>> {
        self.entities
            .iter()
            .cloned()
            .map(TreeItem::from)
            .collect::<Vec<_>>()
    }
}

// ---------- TESTS START HERE

#[cfg(test)]
fn get_file_path(filename: &str) -> PathBuf {
    // cargo sets where project root is
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(manifest_dir).join(filename)
}

#[test]
fn load_nexus_file() {
    let filepath = get_file_path("tests/simple_nexus.h5");
    assert!(filepath.exists());

    // load the nexus file and perform test on root
    let filehandle = FileInfo::read(filepath).unwrap();
    assert!(filehandle.name.ends_with("simple_nexus.h5"));
    assert_eq!(filehandle.size, 45656); // observed

    // other attempt at the tree
    assert_eq!(filehandle.entities.len(), 2); // root node and links
                                              //println!("{:?}", filehandle.entities[0]);
                                              // let entry = GroupInfo::from(filehandle.entities[0]);
                                              //assert_eq!(filehandle.entities[0]["name"], "entry");

    // get to the tree
    let filetree = filehandle.to_tree_items();
    assert_eq!(filetree.len(), 2); // root node and links
}
