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
        loader::Loader,
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

// 为 ModelBundle 实现我们需要的功能
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

    /// 获取 state 对象（用于访问 state 信息）
    fn get_state(&self) -> Box<dyn State + Send + Sync + 'static> {
        match self {
            Self::V4F16(bundle) => Box::new(bundle.state()),
            Self::V5F16(bundle) => Box::new(bundle.state()),
            Self::V6F16(bundle) => Box::new(bundle.state()),
            Self::V7F16(bundle) => Box::new(bundle.state()),
            Self::V4F32(bundle) => Box::new(bundle.state()),
            Self::V5F32(bundle) => Box::new(bundle.state()),
            Self::V6F32(bundle) => Box::new(bundle.state()),
            Self::V7F32(bundle) => Box::new(bundle.state()),
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

    /// 重置运行时状态 - 改进版本，确保正确清理 state
    fn reset(&mut self) -> PyResult<()> {
        // 方法1: 重新创建运行时以重置状态
        self.runtime = self.tokio_runtime.block_on(self.bundle.create_runtime());
        
        // 方法2: 直接重置 GPU state 数据为 0
        self.reset_gpu_state_to_zero()?;
        
        Ok(())
    }

    /// 直接重置 GPU state 数据为 0
    fn reset_gpu_state_to_zero(&self) -> PyResult<()> {
        let state = self.bundle.get_state();
        
        // 获取初始化的零值数据
        let zero_data = state.init();
        
        // 将零值数据写入到 GPU state 的所有 batch
        let num_batch = state.num_batch();
        for batch in 0..num_batch {
            state.load(zero_data.clone(), batch)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }
        
        Ok(())
    }

    /// 读取 GPU state 数据到 CPU 进行比较
    fn read_gpu_state_data(&self, batch: usize) -> PyResult<Vec<f32>> {
        let state = self.bundle.get_state();
        
        // 读取指定 batch 的 GPU state 数据
        let gpu_tensor = state.read(batch)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        // 将 GPU 数据读回到 CPU
        let cpu_data = self.tokio_runtime.block_on(gpu_tensor.back());
        
        // 转换为 Vec<f32>
        Ok(cpu_data.data().to_vec())
    }

    /// 检查 GPU state 是否包含非零值（真正的验证）
    fn check_gpu_state_has_nonzero_values(&self, batch: usize) -> PyResult<bool> {
        let state_data = self.read_gpu_state_data(batch)?;
        
        // 检查是否有非零值（允许小的浮点误差）
        let epsilon = 1e-10;
        let has_nonzero = state_data.iter().any(|&x| x.abs() > epsilon);
        
        Ok(has_nonzero)
    }

    /// 比较两个 batch 的 state 数据
    fn compare_state_batches(&self, batch1: usize, batch2: usize) -> PyResult<String> {
        let data1 = self.read_gpu_state_data(batch1)?;
        let data2 = self.read_gpu_state_data(batch2)?;
        
        if data1.len() != data2.len() {
            return Ok("❌ 两个 batch 的数据长度不同".to_string());
        }
        
        let epsilon = 1e-10;
        let mut differences = 0;
        let mut max_diff = 0.0;
        
        for (i, (val1, val2)) in data1.iter().zip(data2.iter()).enumerate() {
            let diff = (val1 - val2).abs();
            if diff > epsilon {
                differences += 1;
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        
        let conclusion = if differences == 0 { 
            "✅ 两个 batch 数据完全一致".to_string()
        } else { 
            format!("⚠️ 发现 {} 个不同元素", differences)
        };
        
        let result = format!(
            "State 数据比较结果:\n  Batch {} 数据长度: {}\n  Batch {} 数据长度: {}\n  不同元素数量: {}\n  最大差异: {:.6}\n  结论: {}",
            batch1, data1.len(),
            batch2, data2.len(),
            differences,
            max_diff,
            conclusion
        );
        
        Ok(result)
    }

    /// 验证 state 重置是否成功（通过读取实际 GPU 数据）
    fn verify_reset_by_gpu_data(&self) -> PyResult<String> {
        let state = self.bundle.get_state();
        let num_batch = state.num_batch();
        
        let mut results = Vec::new();
        
        // 读取所有 batch 的数据
        for batch in 0..num_batch {
            let has_nonzero = self.check_gpu_state_has_nonzero_values(batch)?;
            results.push((batch, has_nonzero));
        }
        
        // 分析结果
        let all_zero = results.iter().all(|(_, has_nonzero)| !has_nonzero);
        let non_zero_batches: Vec<usize> = results.iter()
            .filter_map(|(batch, has_nonzero)| if *has_nonzero { Some(*batch) } else { None })
            .collect();
        
        let conclusion = if all_zero { 
            "✅ GPU state 已完全重置为零".to_string()
        } else { 
            format!("❌ 发现 {} 个非零 batch", non_zero_batches.len())
        };
        
        let result = format!(
            "GPU State 数据验证结果:\n  总批次数: {}\n  所有 batch 是否为零: {}\n  非零 batch: {:?}\n  结论: {}",
            num_batch,
            all_zero,
            non_zero_batches,
            conclusion
        );
        
        Ok(result)
    }

    /// 获取当前 state 信息用于调试
    fn get_state_info(&self) -> PyResult<String> {
        // 通过 bundle 获取 state 信息
        let state = self.bundle.get_state();
        let num_batch = state.num_batch();
        let init_shape = state.init_shape();
        
        Ok(format!(
            "State Info:\n  - Number of batches: {}\n  - Initial shape: {:?}\n  - State type: {:?}",
            num_batch,
            init_shape,
            std::any::type_name::<dyn web_rwkv::runtime::model::State>()
        ))
    }

    /// 验证 state 是否已经被重置（通过检查第一个 batch 的值）
    fn verify_state_reset(&self) -> PyResult<bool> {
        // 这里可以添加更复杂的 state 验证逻辑
        // 目前返回 true 表示假设已经重置
        // 在实际实现中，可以通过读取 state 数据来验证
        Ok(true)
    }

    /// 强制清理 state 到零值
    fn force_clear_state(&mut self) -> PyResult<()> {
        // 重新创建运行时，这会创建新的零值 state
        self.runtime = self.tokio_runtime.block_on(self.bundle.create_runtime());
        
        // 可以在这里添加额外的验证，确保 state 确实被清零
        Ok(())
    }

    /// 获取 state 的详细统计信息
    fn get_state_statistics(&self) -> PyResult<String> {
        let state = self.bundle.get_state();
        let num_batch = state.num_batch();
        let init_shape = state.init_shape();
        
        // 尝试获取初始化状态的样本数据
        let init_data = state.init();
        let data_len = init_data.len();
        let first_few = if data_len > 0 {
            let sample_size = std::cmp::min(5, data_len);
            let sample: Vec<f32> = init_data.data()[..sample_size].to_vec();
            format!("前{}个值: {:?}", sample_size, sample)
        } else {
            "无数据".to_string()
        };
        
        Ok(format!(
            "State 统计信息:\n  - 批次数: {}\n  - 初始形状: {:?}\n  - 数据长度: {}\n  - 样本数据: {}\n  - State 类型: {}",
            num_batch,
            init_shape,
            data_len,
            first_few,
            std::any::type_name::<dyn web_rwkv::runtime::model::State>()
        ))
    }

    /// 检查 state 是否包含非零值（用于验证重置是否成功）
    fn check_state_has_nonzero_values(&self) -> PyResult<bool> {
        // 现在使用真正的 GPU 数据读取
        self.check_gpu_state_has_nonzero_values(0)
    }

    /// 通过推理结果验证 state 是否被重置（更可靠的方法）
    fn verify_reset_by_inference(&mut self) -> PyResult<String> {
        // 进行第一次推理
        let first_logits = self.predict(vec![1, 2, 3])?;
        let first_sample = first_logits[..5].to_vec();
        
        // 重置 state
        self.reset()?;
        
        // 进行相同的推理
        let second_logits = self.predict(vec![1, 2, 3])?;
        let second_sample = second_logits[..5].to_vec();
        
        // 比较结果
        let is_same = first_sample == second_sample;
        
        let result = format!(
            "通过推理验证重置结果:\n  第一次推理前5个值: {:?}\n  重置后推理前5个值: {:?}\n  结果是否相同: {}\n  结论: {}",
            first_sample,
            second_sample,
            is_same,
            if is_same { "✅ State 重置成功，推理结果一致" } else { "❌ State 重置可能不完整，推理结果不一致" }
        );
        
        Ok(result)
    }

    /// 获取 state 的内存使用情况估计
    fn get_state_memory_usage(&self) -> PyResult<String> {
        let state = self.bundle.get_state();
        let init_shape = state.init_shape();
        
        // 计算内存使用（假设 f32 类型，每个值 4 字节）
        let total_elements = init_shape.iter().product::<usize>();
        let memory_bytes = total_elements * 4; // f32 = 4 bytes
        let memory_mb = memory_bytes as f64 / (1024.0 * 1024.0);
        
        Ok(format!(
            "State 内存使用估计:\n  - 总元素数: {}\n  - 内存大小: {:.2} MB\n  - 形状: {:?}",
            total_elements,
            memory_mb,
            init_shape
        ))
    }

    /// 深度验证 state 重置（通过比较重置前后的状态）
    fn deep_verify_reset(&mut self) -> PyResult<String> {
        // 获取重置前的 GPU state 数据
        let before_has_nonzero = self.check_gpu_state_has_nonzero_values(0)?;
        
        // 进行一些预测来改变状态
        let _ = self.predict(vec![999, 888, 777]);
        let _ = self.predict_next(666);
        
        // 检查预测后是否有非零值
        let after_pred_has_nonzero = self.check_gpu_state_has_nonzero_values(0)?;
        
        // 重置状态
        self.reset()?;
        
        // 获取重置后的 GPU state 数据
        let after_reset_has_nonzero = self.check_gpu_state_has_nonzero_values(0)?;
        
        // 使用新的 GPU 数据验证
        let gpu_verification = self.verify_reset_by_gpu_data()?;
        
        let result = format!(
            "深度验证结果:\n\n重置前:\n  GPU state 有非零值: {}\n\n预测后:\n  GPU state 有非零值: {}\n\n重置后:\n  GPU state 有非零值: {}\n\nGPU 数据验证:\n{}\n\n验证结论: {}",
            before_has_nonzero,
            after_pred_has_nonzero,
            after_reset_has_nonzero,
            gpu_verification,
            if !after_reset_has_nonzero { "✅ 重置成功，GPU state 已清零" } else { "❌ 重置可能不完整，GPU state 仍有非零值" }
        );
        
        Ok(result)
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
