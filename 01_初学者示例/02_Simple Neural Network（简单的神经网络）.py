"""
从零开始构建简单的神经网络（Neural Network）
===================================

这个例子演示了如何在不依赖任何机器学习框架的情况下手写一个神经网络。
这将帮助你透彻理解神经网络“底层”的运行机制。

你将学到：
- 神经元（neurons）的工作原理
- 前向传播（Forward propagation）（做出预测）
- 反向传播（Backward propagation）（从错误中学习）
- 什么是 Sigmoid 激活函数（用于控制信号）

实战案例：让 AI 学会判断点是在直线的上面还是下面。
"""

import random
import math

def sigmoid(x):
    """
    Sigmoid 激活函数：将任意数值转换为 0 到 1 之间的数。
    
    这就好比在问AI：“你有多大把握？” 
    - 输出接近 1 ：AI说“我非常有信心，是！”
    - 输出接近 0 ：AI说“我非常有信心，不是！”
    - 输出接近 0.5 ：AI说“额……我不确定。”
    
    参数：
        x：输入值
        
    返回值：
        0 到 1 之间的数值
    """
    # 针对极大或极小数值，防止计算溢出
    if x > 100:
        return 1.0
    if x < -100:
        return 0.0
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    """
    Sigmoid 函数的导数（derivative）——学习过程必不可少。
    它告诉我们需要对权重进行多大的调整。
    
    参数：
        x: Sigmoid 函数的输出值
        
    返回值：
        导数值
    """
    return x * (1 - x)


class SimpleNeuron:
    """
    单个“人工神经元”——这是构建神经网络的基石。
    
    你可以把它想象成一个微型的“决策器”，它的工作流程是这样的：
    1. 接收输入（比如数据的各种特征）
    2. 将它们乘以AI学到的权重 
    3. 加上“偏置”（bias）并求和
    4. 通过“激活函数”处理
    5. 输出一个预测结果
    """
    
    def __init__(self, num_inputs):
        """
        用随机权重初始化这个神经元。
        
        参数:
            num_inputs: 这个神经元将要接收多少个输入值
        """
        # 每个输入都会有一个对应的权重（用来判断这个输入到底有多重要？）
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        # 偏置用来微调最终的输出结果
        self.bias = random.uniform(-1, 1)
        # 保存最后的输出，以便学习阶段调用
        self.output = 0
        
    def feedforward(self, inputs):
        """
        计算神经元的输出（即预测值）。
        这个过程就叫做“前向传播”（forward propagation）。
        
        参数:
            inputs: 输入值的列表
            
        返回值:
            神经元的输出结果（数值介于 0 到 1 之间）
        """
        # 步骤一: 把每个输入值与其权重相乘，再求总和
        total = sum(w * x for w, x in zip(self.weights, inputs))
        
        # 步骤二: 加上偏置值
        total += self.bias
        
        # 步骤三: 使用 Sigmoid 函数进行激活处理
        self.output = sigmoid(total)
        
        return self.output
    
    def train(self, inputs, target, learning_rate=0.1):
        """
        教这个神经元优化它的预测能力。
        这个过程就叫做“反向传播”（backward propagation）。
        
        参数:
            inputs: The input values
            target: What the output should have been
            learning_rate: How much to adjust weights
        """
        # 计算误差（error）
        error = target - self.output
        
        # 利用导数来计算调整量（delta）
        delta = error * sigmoid_derivative(self.output)
        
        # 更新权重（weights）
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * delta * inputs[i]
        
        # 更新偏置（bias）
        self.bias += learning_rate * delta
        
        return abs(error)


def generate_training_data(num_samples=100):
    """
    Generate sample data for training.
    
    Task: Classify points as above (1) or below (0) the line y = x.
    
    Args:
        num_samples: How many training examples to create
        
    Returns:
        List of (inputs, target) tuples
    """
    data = []
    for _ in range(num_samples):
        # Random point in 2D space (x, y coordinates)
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        
        # Label: 1 if point is above the line y=x, 0 if below
        label = 1 if y > x else 0
        
        data.append(([x, y], label))
    
    return data


def visualize_decision(neuron, test_points):
    """
    Show how the neuron classifies different points.
    
    Args:
        neuron: Trained neuron
        test_points: List of points to test
    """
    print("\n🎯 Testing the trained neuron:")
    print("-" * 70)
    print(f"{'Point':<15} | {'Prediction':<15} | {'Actual':<15} | {'Correct?'}")
    print("-" * 70)
    
    correct = 0
    for point, actual in test_points:
        prediction = neuron.feedforward(point)
        predicted_class = 1 if prediction > 0.5 else 0
        actual_class = actual
        is_correct = "✓" if predicted_class == actual_class else "✗"
        
        if predicted_class == actual_class:
            correct += 1
        
        print(f"({point[0]:5.2f}, {point[1]:5.2f}) | {prediction:14.4f} | {actual_class:^15} | {is_correct}")
    
    print("-" * 70)
    accuracy = (correct / len(test_points)) * 100
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(test_points)} correct)")


def main():
    """
    Main function - Build and train a neural network!
    """
    print("=" * 70)
    print("Simple Neural Network from Scratch")
    print("=" * 70)
    print("\n📚 Task: Learn to classify points as above or below the line y = x")
    print()
    
    # Step 1: Generate training data
    print("📊 Generating training data...")
    training_data = generate_training_data(num_samples=100)
    print(f"Created {len(training_data)} training examples")
    
    # Show a few examples
    print("\nExample training data:")
    for i in range(3):
        point, label = training_data[i]
        position = "above" if label == 1 else "below"
        print(f"  Point ({point[0]:.2f}, {point[1]:.2f}) is {position} the line y=x")
    
    # Step 2: Create neuron
    print("\n🧠 Creating a neuron with 2 inputs (x and y coordinates)...")
    neuron = SimpleNeuron(num_inputs=2)
    print(f"Initial weights: [{neuron.weights[0]:.3f}, {neuron.weights[1]:.3f}]")
    print(f"Initial bias: {neuron.bias:.3f}")
    
    # Step 3: Train the neuron
    print("\n🎓 Training the neuron...")
    epochs = 50
    
    for epoch in range(epochs):
        total_error = 0
        
        # Train on each example
        for inputs, target in training_data:
            neuron.feedforward(inputs)
            error = neuron.train(inputs, target, learning_rate=0.1)
            total_error += error
        
        # Show progress
        if (epoch + 1) % 10 == 0:
            avg_error = total_error / len(training_data)
            print(f"Epoch {epoch + 1}/{epochs} - Average error: {avg_error:.4f}")
    
    print("\n✅ Training complete!")
    print(f"Final weights: [{neuron.weights[0]:.3f}, {neuron.weights[1]:.3f}]")
    print(f"Final bias: {neuron.bias:.3f}")
    
    # Step 4: Test the neuron
    test_data = generate_training_data(num_samples=10)
    visualize_decision(neuron, test_data)
    
    # Explanation
    print("\n💡 What just happened?")
    print("1. The neuron started with random weights")
    print("2. It looked at 100 example points and their correct labels")
    print("3. Each time it was wrong, it adjusted its weights slightly")
    print("4. After 50 rounds, it learned to classify points correctly!")
    print()
    print("🎉 You just built a neural network from scratch!")
    print()
    print("🚀 Try this:")
    print("   - Change num_samples to train on more/fewer examples")
    print("   - Modify epochs to train for longer/shorter")
    print("   - Change learning_rate (line 185) and see what happens")
    print("   - Try different decision boundaries (modify generate_training_data)")
    print()


if __name__ == "__main__":
    main()