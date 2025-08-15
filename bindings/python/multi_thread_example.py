#!/usr/bin/env python3
"""
WebRWKV 多线程使用示例 - 新架构版本

这个示例展示了如何：
1. 在多线程间共享同一个模型参数
2. 每个线程拥有独立的推理运行时
3. 避免重复加载模型参数
4. 实现高效的并发推理
5. 正确处理状态隔离
"""

import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from webrwkv_py import Model, ThreadRuntime, get_available_adapters_py

class SharedModelManager:
    """共享模型管理器 - 实现多线程共享模型参数"""
    
    def __init__(self, model_path: str, precision: str = "fp16", adapter_index: int = None):
        """
        初始化共享模型管理器
        
        Args:
            model_path: 模型文件路径
            precision: 精度设置
            adapter_index: GPU设备索引
        """
        print(f"🔧 正在加载共享模型: {model_path}")
        print(f"📊 精度: {precision}")
        
        # 只加载一次模型参数
        self.shared_model = Model(model_path, precision, adapter_index)
        self.model_path = model_path
        self.precision = precision
        self.adapter_index = adapter_index
        
        print("✅ 共享模型加载完成！")
        print(f"📁 模型路径: {self.shared_model.get_model_path()}")
        print(f"⚙️  精度: {self.shared_model.get_precision()}")
    
    def create_thread_runtime(self) -> 'ThreadModel':
        """
        为每个线程创建独立的运行时
        
        Returns:
            ThreadModel: 线程专用的模型包装器
        """
        # 创建线程专用的推理运行时
        thread_runtime = self.shared_model.create_thread_runtime()
        return ThreadModel(self.model_path, self.precision, self.shared_model, thread_runtime)
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "precision": self.precision,
            "adapter_index": self.adapter_index
        }

class ThreadModel:
    """线程专用的模型包装器 - 共享模型参数但独立运行时"""
    
    def __init__(self, model_path: str, precision: str, shared_model: Model, thread_runtime: ThreadRuntime):
        """
        初始化线程模型
        
        Args:
            model_path: 模型文件路径
            precision: 精度设置
            shared_model: 共享的模型实例
            thread_runtime: 线程专用的推理运行时
        """
        self.model_path = model_path
        self.precision = precision
        # 共享模型参数
        self.shared_model = shared_model
        # 每个线程独立的运行时
        self.thread_runtime = thread_runtime
        
        # 每个线程都有独立的运行时状态
        self.thread_id = threading.get_ident()
        print(f"🧵 线程 {self.thread_id} 创建独立运行时")
        print(f"  模型路径: {self.model_path}")
        print(f"  精度: {self.precision}")
    
    def predict(self, ids: list[int]) -> list[float]:
        """使用独立运行时进行预测"""
        print(f"🧵 线程 {self.thread_id} 开始预测: {ids}")
        
        # 使用线程专用的运行时进行预测
        logits = self.thread_runtime.predict(ids)
        
        print(f"🧵 线程 {self.thread_id} 预测完成，输出 {len(logits)} 个logits")
        return logits
    
    def predict_next(self, token_id: int) -> list[float]:
        """使用独立运行时进行增量预测"""
        print(f"⏭️ 线程 {self.thread_id} 增量预测: {token_id}")
        
        # 使用线程专用的运行时进行增量预测
        logits = self.thread_runtime.predict_next(token_id)
        
        print(f"🧵 线程 {self.thread_id} 增量预测完成")
        return logits
    
    def reset(self):
        """重置线程的运行时状态"""
        print(f"🔄 线程 {self.thread_id} 重置运行时")
        self.thread_runtime.reset()
    
    def get_thread_info(self):
        """获取线程信息"""
        return {
            "thread_id": self.thread_id,
            "model_path": self.model_path,
            "precision": self.precision
        }

def worker_task(thread_id: int, model: ThreadModel, input_data: list[int]):
    """
    工作线程任务
    
    Args:
        thread_id: 线程ID
        model: 线程专用的模型包装器
        input_data: 输入数据
    """
    try:
        print(f"🚀 线程 {thread_id} 开始工作")
        print(f"  线程ID: {model.thread_id}")
        print(f"  模型路径: {model.model_path}")
        
        # 每个线程独立进行推理
        logits = model.predict(input_data)
        
        # 模拟一些处理时间
        time.sleep(0.1)
        
        # 增量推理
        next_logits = model.predict_next(42)
        
        # 重置状态
        model.reset()
        
        print(f"✅ 线程 {thread_id} 工作完成")
        return {
            "thread_id": thread_id,
            "thread_ident": model.thread_id,
            "input": input_data,
            "logits_count": len(logits),
            "next_logits_count": len(next_logits)
        }
        
    except Exception as e:
        print(f"❌ 线程 {thread_id} 出错: {e}")
        return {"thread_id": thread_id, "error": str(e)}

def demonstrate_multi_threading():
    """演示多线程使用"""
    print("🚀 WebRWKV 多线程使用演示 - 新架构版本")
    print("=" * 60)
    
    # 1. 显示可用设备
    print("🎮 可用的GPU设备:")
    adapters = get_available_adapters_py()
    for index, name in adapters:
        print(f"  {index}: {name}")
    
    # 2. 选择设备
    selected_adapter = None
    for index, name in adapters:
        if "Vulkan" in name:
            selected_adapter = index
            break
    
    print(f"\n🎯 选择设备: {selected_adapter}")
    
    # 3. 创建共享模型管理器
    model_path = "/home/yueyulin/github/web-rwkv/rwkv_tts.st"
    
    try:
        shared_manager = SharedModelManager(
            model_path=model_path,
            precision="fp32",  # 使用 fp32 精度
            adapter_index=selected_adapter
        )
        
        print(f"\n📊 共享模型管理器信息: {shared_manager.get_model_info()}")
        
        # 4. 多线程测试
        print(f"\n🧪 开始多线程测试")
        
        # 准备测试数据
        test_inputs = [
            [1, 2, 3, 4, 5],      # 线程1的输入
            [10, 20, 30, 40, 50], # 线程2的输入
            [100, 200, 300, 400, 500], # 线程3的输入
            [1000, 2000, 3000, 4000, 5000], # 线程4的输入
        ]
        
        # 使用线程池执行
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 为每个线程创建独立的运行时
            futures = []
            for i, input_data in enumerate(test_inputs):
                thread_model = shared_manager.create_thread_runtime()
                future = executor.submit(worker_task, i+1, thread_model, input_data)
                futures.append(future)
            
            # 收集结果
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"📋 结果: {result}")
        
        print(f"\n✅ 多线程测试完成！")
        print(f"📊 总结果数量: {len(results)}")
        
        # 5. 验证独立性
        print(f"\n🔍 验证线程独立性:")
        for result in results:
            if "error" not in result:
                print(f"  线程 {result['thread_id']}: ID {result['thread_ident']}")
        
        # 创建新的线程运行时验证
        new_thread_model = shared_manager.create_thread_runtime()
        print(f"  新线程模型信息: {new_thread_model.get_thread_info()}")
        
    except Exception as e:
        print(f"❌ 多线程测试失败: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_state_independence():
    """演示状态独立性"""
    print(f"\n🔬 状态独立性演示")
    print("=" * 60)
    
    try:
        # 创建共享模型管理器
        model_path = "/home/yueyulin/github/web-rwkv/rwkv_tts.st"
        shared_manager = SharedModelManager(model_path, "fp32")
        
        # 创建两个线程运行时
        model1 = shared_manager.create_thread_runtime()
        model2 = shared_manager.create_thread_runtime()
        
        print("🧵 创建两个独立的线程运行时")
        print(f"  模型1线程ID: {model1.thread_id}")
        print(f"  模型2线程ID: {model2.thread_id}")
        
        # 线程1进行推理
        print(f"\n🧵 线程1开始推理:")
        logits1 = model1.predict([1, 2, 3])
        print(f"前 5 个logits: {logits1[:5]}")
        next_logits1 = model1.predict_next(10)
        print(f"  线程1 logits数量: {len(logits1)}")
        print(f"  线程1 增量logits数量: {len(next_logits1)}")
        
        # 线程2进行推理（应该有不同的状态）
        print(f"\n🧵 线程2开始推理:")
        logits2 = model2.predict([100, 200, 300])
        print(f"前 5 个logits: {logits2[:5]}")
        next_logits2 = model2.predict_next(1000)
        print(f"  线程2 logits数量: {len(logits2)}")
        print(f"前 5 个logits: {logits2[:5]}")
        print(f"  线程2 增量logits数量: {len(next_logits2)}")
        
        # 重置线程1的状态
        print(f"\n🔄 重置线程1状态:")
        model1.reset()
        
        # 验证线程2仍然可以正常工作
        print(f"\n🧪 验证线程2独立性:")
        logits2_after = model2.predict([500, 600, 700])
        print(f"前 5 个logits: {logits2_after[:5]}")
        print(f"  线程2重置后仍可工作，logits数量: {len(logits2_after)}")
        
        print("✅ 状态独立性验证完成！")
        
    except Exception as e:
        print(f"❌ 状态独立性演示失败: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_memory_efficiency():
    """演示内存效率"""
    print(f"\n💾 内存效率演示")
    print("=" * 60)
    
    try:
        import psutil
        import os
        
        def get_memory_usage():
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # MB
        
        # 创建共享模型管理器
        model_path = "/home/yueyulin/github/web-rwkv/rwkv_tts.st"
        
        print("📊 测试内存使用:")
        initial_memory = get_memory_usage()
        print(f"  初始内存: {initial_memory:.2f} MB")
        
        shared_manager = SharedModelManager(model_path, "fp32")
        after_model_load = get_memory_usage()
        print(f"  加载模型后: {after_model_load:.2f} MB")
        print(f"  模型加载内存增长: {after_model_load - initial_memory:.2f} MB")
        
        # 创建多个线程运行时
        thread_models = []
        for i in range(4):
            thread_model = shared_manager.create_thread_runtime()
            thread_models.append(thread_model)
            
            current_memory = get_memory_usage()
            memory_growth = current_memory - after_model_load
            print(f"  创建线程运行时 {i+1}: 内存增长 {memory_growth:.2f} MB")
        
        final_memory = get_memory_usage()
        total_growth = final_memory - initial_memory
        model_growth = after_model_load - initial_memory
        
        print(f"\n📈 内存使用分析:")
        print(f"  模型参数内存: {model_growth:.2f} MB")
        print(f"  总内存增长: {total_growth:.2f} MB")
        print(f"  运行时内存: {total_growth - model_growth:.2f} MB")
        
        if total_growth < model_growth * 2:
            print(f"  ✅ 内存效率良好！模型参数被正确共享")
        else:
            print(f"  ⚠️  内存效率一般，可能存在重复加载")
        
        # 清理
        del thread_models
        del shared_manager
        
        print("✅ 内存效率演示完成！")
        
    except ImportError:
        print("⚠️  psutil 未安装，跳过内存测试")
    except Exception as e:
        print(f"❌ 内存效率演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行多线程演示
    demonstrate_multi_threading()
    
    # 运行状态独立性演示
    demonstrate_state_independence()
    
    # 运行内存效率演示
    demonstrate_memory_efficiency()
    
    print(f"\n🎉 所有演示完成！")
