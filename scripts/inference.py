"""
Optimized GPU Inference with CUDA Streams
Implements concurrent execution for improved throughput
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np

# Import model architecture
import sys
sys.path.append('.')
from train import CIFAR10CNN

def profile_inference(model, test_loader, device, use_streams=False):
    """
    Profile inference performance with optional CUDA streams
    """
    model.eval()
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    correct = 0
    total = 0
    inference_times = []
    
    if use_streams and torch.cuda.is_available():
        # Create CUDA streams for concurrent execution
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        streams = [stream1, stream2]
        stream_idx = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            batch_start = time.time()
            
            if use_streams and torch.cuda.is_available():
                # Use alternating streams for concurrent execution
                current_stream = streams[stream_idx % 2]
                with torch.cuda.stream(current_stream):
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    
                    # Synchronize stream
                    current_stream.synchronize()
                
                stream_idx += 1
            else:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
            
            batch_time = time.time() - batch_start
            inference_times.append(batch_time)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Print sample predictions
            if batch_idx == 0:
                print(f'\n[v0] Sample predictions from first batch:')
                for i in range(min(5, len(predicted))):
                    print(f'[v0] Predicted: {classes[predicted[i]]}, Actual: {classes[targets[i]]}')
    
    accuracy = 100. * correct / total
    avg_time = np.mean(inference_times)
    throughput = len(test_loader.dataset) / sum(inference_times)
    
    return accuracy, avg_time, throughput, inference_times

def benchmark_memory():
    """
    Profile GPU memory usage
    """
    if torch.cuda.is_available():
        print(f'\n[v0] GPU Memory Profiling:')
        print(f'[v0] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
        print(f'[v0] Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB')
        print(f'[v0] Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB')

def main():
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[v0] Using device: {device}')
    
    # Load test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                    download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, 
                            shuffle=False, num_workers=4, 
                            pin_memory=True)
    
    # Load model
    print('[v0] Loading trained model...')
    model = CIFAR10CNN(num_classes=10).to(device)
    
    try:
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'[v0] Loaded model with training accuracy: {checkpoint["accuracy"]:.2f}%')
    except FileNotFoundError:
        print('[v0] Warning: best_model.pth not found, using untrained model')
    
    # Benchmark without CUDA streams
    print('\n[v0] === Baseline Inference (No CUDA Streams) ===')
    acc1, avg_time1, throughput1, _ = profile_inference(model, test_loader, device, use_streams=False)
    print(f'[v0] Accuracy: {acc1:.2f}%')
    print(f'[v0] Average Batch Time: {avg_time1*1000:.2f} ms')
    print(f'[v0] Throughput: {throughput1:.2f} images/sec')
    benchmark_memory()
    
    # Benchmark with CUDA streams
    if torch.cuda.is_available():
        print('\n[v0] === Optimized Inference (With CUDA Streams) ===')
        torch.cuda.reset_peak_memory_stats()
        acc2, avg_time2, throughput2, _ = profile_inference(model, test_loader, device, use_streams=True)
        print(f'[v0] Accuracy: {acc2:.2f}%')
        print(f'[v0] Average Batch Time: {avg_time2*1000:.2f} ms')
        print(f'[v0] Throughput: {throughput2:.2f} images/sec')
        benchmark_memory()
        
        # Calculate speedup
        speedup = ((avg_time1 - avg_time2) / avg_time1) * 100
        print(f'\n[v0] === Performance Improvement ===')
        print(f'[v0] Inference Speedup: {speedup:.2f}%')
        print(f'[v0] Throughput Increase: {((throughput2 - throughput1) / throughput1) * 100:.2f}%')
    
    # Export model for deployment
    print('\n[v0] Exporting model for deployment...')
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    torch.onnx.export(model, dummy_input, 'model.onnx', 
                     export_params=True, opset_version=11,
                     input_names=['input'], output_names=['output'])
    print('[v0] Model exported to model.onnx')

if __name__ == '__main__':
    main()
