#!/usr/bin/env python3
"""
GPU State 重置和读取测试脚本
测试真正的 GPU state 数据操作
"""

import webrwkv_py
import numpy as np
import os

def test_gpu_state_operations():
    """测试 GPU state 操作"""
    print("🚀 开始 GPU State 操作测试")
    print("=" * 50)
    
    try:
        # 检查模型文件是否存在
        model_path = "/home/yueyulin/models/rwkv7-0.4B-g1-respark-voice-tunable_ipa_10k/webrwkv.safetensors"
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print("请确保模型文件路径正确")
            return
        
        # 创建模型
        print("📦 创建 Model...")
        model = webrwkv_py.Model(model_path, "fp32", None)
        print("✅ Model 创建成功")
        
        # 创建线程运行时
        print("📦 创建 ThreadRuntime...")
        runtime = model.create_thread_runtime()
        print("✅ ThreadRuntime 创建成功")
        
        # 获取初始状态信息
        print("\n📊 获取初始状态信息:")
        state_info = runtime.get_state_info()
        print(state_info)
        
        # 检查初始状态是否有非零值
        print("\n🔍 检查初始状态是否有非零值:")
        has_nonzero = runtime.check_state_has_nonzero_values()
        print(f"初始状态有非零值: {has_nonzero}")
        
        # 进行一些预测来改变状态
        print("\n🎯 进行预测来改变状态...")
        tokens = [100, 200, 300, 400, 500]
        for token in tokens:
            logits = runtime.predict([token])
            print(f"输入 token {token}, 输出 logits 前3个值: {logits[:3]}")
        
        # 检查预测后状态是否有非零值
        print("\n🔍 检查预测后状态是否有非零值:")
        after_pred_has_nonzero = runtime.check_state_has_nonzero_values()
        print(f"预测后状态有非零值: {after_pred_has_nonzero}")
        
        # 读取 GPU state 数据
        print("\n📖 读取 GPU state 数据 (batch 0):")
        try:
            state_data = runtime.read_gpu_state_data(0)
            print(f"GPU state 数据长度: {len(state_data)}")
            print(f"前10个值: {state_data[:10]}")
            print(f"后10个值: {state_data[-10:]}")
            
            # 检查非零值
            non_zero_count = sum(1 for x in state_data if abs(x) > 1e-10)
            print(f"非零值数量: {non_zero_count}")
            print(f"非零值比例: {non_zero_count/len(state_data)*100:.2f}%")
            
        except Exception as e:
            print(f"❌ 读取 GPU state 数据失败: {e}")
        
        # 重置状态
        print("\n🔄 重置状态...")
        runtime.reset()
        print("✅ 状态重置完成")
        
        # 检查重置后状态是否有非零值
        print("\n🔍 检查重置后状态是否有非零值:")
        after_reset_has_nonzero = runtime.check_state_has_nonzero_values()
        print(f"重置后状态有非零值: {after_reset_has_nonzero}")
        
        # 使用新的 GPU 数据验证
        print("\n🔍 使用 GPU 数据验证重置:")
        try:
            gpu_verification = runtime.verify_reset_by_gpu_data()
            print(gpu_verification)
        except Exception as e:
            print(f"❌ GPU 数据验证失败: {e}")
        
        # 比较重置前后的状态
        print("\n📊 深度验证重置:")
        try:
            deep_verification = runtime.deep_verify_reset()
            print(deep_verification)
        except Exception as e:
            print(f"❌ 深度验证失败: {e}")
        
        # 比较不同 batch 的状态
        print("\n🔍 比较不同 batch 的状态:")
        try:
            batch_comparison = runtime.compare_state_batches(0, 0)  # 同一个 batch
            print(batch_comparison)
        except Exception as e:
            print(f"❌ Batch 比较失败: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 GPU State 操作测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu_state_operations()
