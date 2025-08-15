use anyhow::Result;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime as TokioRt;

// Import necessary items from the web-rwkv crate
use web_rwkv::{
    context::{ContextBuilder, InstanceExt},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::{Loader, Reader},
        model::{ModelBuilder, ModelVersion, State, Bundle},
        v4, v5, v6, v7, TokioRuntime,
    },
};

// Import half for f16 type
use half::f16;

// An internal enum to hold a bundle of a specific precision and version.
#[derive(Clone)]  // 添加 Clone trait
enum ModelBundle {
    V4F16(v4::Bundle<f16>),
    V5F16(v5::Bundle<f16>),
    V6F16(v6::Bundle<f16>),
    V7F16(v7::Bundle<f16>),
    V4F32(v4::Bundle<f32>),
    V5F32(v5::Bundle<f32>),
    V6F32(v6::Bundle<f32>),
    V7F32(v7::Bundle<f32>),
}

impl ModelBundle {
    /// Create a TokioRuntime from the bundle (reuse the same runtime)
    async fn create_runtime(&self) -> TokioRuntime<Rnn> {
        match self {
            Self::V4F16(bundle) => TokioRuntime::new(bundle.clone()).await,
            Self::V5F16(bundle) => TokioRuntime::new(bundle.clone()).await,
            Self::V6F16(bundle) => TokioRuntime::new(bundle.clone()).await,
            Self::V7F16(bundle) => TokioRuntime::new(bundle.clone()).await,
            Self::V4F32(bundle) => TokioRuntime::new(bundle.clone()).await,
            Self::V5F32(bundle) => TokioRuntime::new(bundle.clone()).await,
            Self::V6F32(bundle) => TokioRuntime::new(bundle.clone()).await,
            Self::V7F32(bundle) => TokioRuntime::new(bundle.clone()).await,
        }
    }
}

#[pyclass]
struct Model {
    bundle: ModelBundle,
    tokio_runtime: Arc<TokioRt>,
    // Store original paths to allow for resetting.
    model_path: PathBuf,
    precision: String,
    // 🚨 移除共享的 runtime 字段，改为每个线程独立管理
}

/// Helper function to load a bundle. Used by `new` and `reset`.
async fn load_bundle(model_path: &PathBuf, precision: &str, adapter_index: Option<usize>) -> Result<ModelBundle> {
    let file = tokio::fs::File::open(model_path).await?;
    let data = unsafe { memmap2::Mmap::map(&file)? };
    let model_tensors = safetensors::SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model_tensors)?;

    let instance = wgpu::Instance::default();
    
    // Handle device selection
    let adapter = if let Some(index) = adapter_index {
        let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).into_iter().collect();
        if index >= adapters.len() {
            return Err(anyhow::anyhow!("Invalid adapter index: {}. Available adapters: {}", index, adapters.len()));
        }
        adapters[index].clone()
    } else {
        instance.adapter(wgpu::PowerPreference::HighPerformance).await?
    };
    
    let limits = adapter.limits();
    let context = ContextBuilder::new(adapter).limits(limits).build().await?;

    let builder = ModelBuilder::new(&context, model_tensors);

    match precision.to_lowercase().as_str() {
        "fp16" => {
            let bundle = match info.version {
                ModelVersion::V4 => ModelBundle::V4F16(v4::Bundle::new(builder.build_v4().await?, 1)),
                ModelVersion::V5 => ModelBundle::V5F16(v5::Bundle::new(builder.build_v5().await?, 1)),
                ModelVersion::V6 => ModelBundle::V6F16(v6::Bundle::new(builder.build_v6().await?, 1)),
                ModelVersion::V7 => ModelBundle::V7F16(v7::Bundle::new(builder.build_v7().await?, 1)),
            };
            Ok(bundle)
        }
        "fp32" => {
            let bundle = match info.version {
                ModelVersion::V4 => ModelBundle::V4F32(v4::Bundle::new(builder.build_v4().await?, 1)),
                ModelVersion::V5 => ModelBundle::V5F32(v5::Bundle::new(builder.build_v5().await?, 1)),
                ModelVersion::V6 => ModelBundle::V6F32(v6::Bundle::new(builder.build_v6().await?, 1)),
                ModelVersion::V7 => ModelBundle::V7F32(v7::Bundle::new(builder.build_v7().await?, 1)),
            };
            Ok(bundle)
        }
        _ => Err(anyhow::anyhow!("Unsupported precision: {}. Use 'fp16' or 'fp32'", precision))
    }
}

/// Get list of available GPU adapters
fn get_available_adapters() -> Vec<(usize, String)> {
    let instance = wgpu::Instance::default();
    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).into_iter().collect();
    
    adapters.into_iter().enumerate().map(|(index, adapter)| {
        let info = adapter.get_info();
        (index, format!("{} ({:?})", info.name, info.backend))
    }).collect()
}

#[pyclass]
struct ThreadRuntime {
    runtime: TokioRuntime<Rnn>,
    tokio_runtime: Arc<TokioRt>,
    bundle: ModelBundle,  // 添加bundle引用
}

#[pymethods]
impl ThreadRuntime {
    /// 使用独立运行时进行预测
    fn predict(&mut self, ids: Vec<u32>) -> PyResult<Vec<f32>> {
        if ids.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Input ids cannot be empty"));
        }
        
        let input = RnnInput::new(vec![RnnInputBatch::new(ids, RnnOption::Last)], 128);
        let logits = self.tokio_runtime.block_on(async {
            let mut inference = input;
            loop {
                let (next_inference, output) = self.runtime.infer(inference).await
                    .map_err(anyhow::Error::from)?;
                inference = next_inference;
                if !output.is_empty() && !output[0].is_empty() {
                    return Ok(output[0].0.to_vec());
                }
            }
        }).map_err(|e: anyhow::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(logits)
    }

    /// 使用独立运行时进行增量预测
    fn predict_next(&mut self, token_id: u32) -> PyResult<Vec<f32>> {
        let input = RnnInput::new(vec![RnnInputBatch::new(vec![token_id], RnnOption::Last)], 128);
        let logits = self.tokio_runtime.block_on(async {
            let mut inference = input;
            loop {
                let (next_inference, output) = self.runtime.infer(inference).await
                    .map_err(anyhow::Error::from)?;
                inference = next_inference;
                if !output.is_empty() && !output[0].is_empty() {
                    return Ok(output[0].0.to_vec());
                }
            }
        }).map_err(|e: anyhow::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(logits)
    }

    /// 重置运行时状态
    fn reset(&mut self) -> PyResult<()> {
        // 重新创建运行时以重置状态
        self.runtime = self.tokio_runtime.block_on(self.bundle.create_runtime());
        Ok(())
    }
}

#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (model_path, precision, adapter_index=None))]
    fn new(model_path: PathBuf, precision: String, adapter_index: Option<usize>) -> PyResult<Self> {
        let tokio_runtime = Arc::new(TokioRt::new()?);
        let bundle = tokio_runtime.block_on(load_bundle(&model_path, &precision, adapter_index))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { 
            bundle, 
            tokio_runtime, 
            model_path, 
            precision,
        })
    }

    /// 创建线程专用的推理运行时
    fn create_thread_runtime(&self) -> PyResult<ThreadRuntime> {
        let runtime = self.tokio_runtime.block_on(self.bundle.create_runtime());
        Ok(ThreadRuntime {
            runtime,
            tokio_runtime: self.tokio_runtime.clone(),
            bundle: self.bundle.clone(),  // 克隆bundle
        })
    }

    /// 获取当前精度设置
    fn get_precision(&self) -> &str {
        &self.precision
    }

    /// 获取模型路径
    fn get_model_path(&self) -> &str {
        self.model_path.to_str().unwrap_or("Invalid path")
    }
}

/// 获取可用的GPU适配器列表
#[pyfunction]
fn get_available_adapters_py() -> Vec<(usize, String)> {
    get_available_adapters()
}

#[pymodule]
fn webrwkv_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<ThreadRuntime>()?;
    m.add_function(wrap_pyfunction!(get_available_adapters_py, m)?)?;
    Ok(())
}
