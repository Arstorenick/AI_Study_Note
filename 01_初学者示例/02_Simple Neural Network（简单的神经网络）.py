"""
从零开始构建简单的神经网络(Neural Network)
===================================

这个例子演示了如何在不依赖任何机器学习框架的情况下手写一个神经网络。
这将帮助你透彻理解神经网络“底层”的运行机制。

你将学到：
- 神经元(neurons)的工作原理
- 前向传播(Forward propagation)(做出预测)
- 反向传播(Backward propagation)(从错误中学习)
- 什么是 Sigmoid 激活函数(用于控制信号)

应用案例：让 AI 学会判断点是在直线的上面还是下面。
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
            inputs: 输入的数据值
            target: 理想的输出结果
            learning_rate: 权重的调整幅度
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
    生成用于训练的样本数据。
    
    任务: 
        将点分类为位于直线 y = x 上方 (1) 或下方 (0)。
    
    参数:
        num_samples: 要创建多少个训练样本。
        
    返回值:
        一个包含 (输入, 目标值) 元组的列表
    """
    data = []
    for _ in range(num_samples):
        # 二维空间中的随机点（x, y 坐标）
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        
        # 标签：如果点在直线 y=x 上方则为 1，在下方则为 0
        label = 1 if y > x else 0
        
        data.append(([x, y], label))
    
    return data


def visualize_decision(neuron, test_points):
    """
    展示神经元是如何对不同点进行归类的。
    
    参数:
        neuron: 已经训练好的神经元
        test_points: 要测试的点列表
    """
    print("\n🎯 正在对训练完毕的神经元进行测试:")
    print("-" * 70)
    print(f"{'点位':<15} | {'预测值':<15} | {'实际值':<15} | {'是否正确?'}")
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
    print(f"准确率: {accuracy:.1f}% ({correct}/{len(test_points)} 正确)")


def main():
    """
    主函数 —— 构建并训练一个神经网络！
    """
    print("=" * 70)
    print("从零开始构建简易简单网络")
    print("=" * 70)
    print("\n📚 任务: 学会将点分类为位于直线 y = x 的上方或下方")
    print()
    
    # 步骤一: 生成训练数据
    print("📊 正在创建训练数据...")
    training_data = generate_training_data(num_samples=100)
    print(f"成功生成了 {len(training_data)} 条训练数据")
    
    # 显示几条样本数据
    print("\n训练数据示例:")
    for i in range(3):
        point, label = training_data[i]
        position = "上面" if label == 1 else "下面"
        print(f"  点 ({point[0]:.2f}, {point[1]:.2f}) 位于直线y=x的 {position}。")
    
    # 步骤二: 创建神经元
    print("\n🧠 正在创建一个具有 2 个输入( x 和 y 坐标)的神经元...")
    neuron = SimpleNeuron(num_inputs=2)
    print(f"初始权重(weights): [{neuron.weights[0]:.3f}, {neuron.weights[1]:.3f}]")
    print(f"初始偏置值(bias): {neuron.bias:.3f}")
    
    # 步骤三: 训练神经元
    print("\n🎓 正在训练神经元...")
    epochs = 50
    
    for epoch in range(epochs):
        total_error = 0
        
        # 逐个样本进行训练
        for inputs, target in training_data:
            neuron.feedforward(inputs)
            error = neuron.train(inputs, target, learning_rate=0.1)
            total_error += error
        
        # 展示进度
        if (epoch + 1) % 10 == 0:
            avg_error = total_error / len(training_data)
            print(f"第 {epoch + 1}/{epochs} 轮 - 平均误差: {avg_error:.4f}")
    
    print("\n✅ 训练完成!")
    print(f"最终权重(weights): [{neuron.weights[0]:.3f}, {neuron.weights[1]:.3f}]")
    print(f"最终偏置值(bias): {neuron.bias:.3f}")
    
    # Step 4: Test the neuron
    test_data = generate_training_data(num_samples=10)
    visualize_decision(neuron, test_data)
    
    # Explanation
    print("\n💡 刚才发生了什么？")
    print("1. 神经元一开始使用的是随机生成的权重")
    print("2. 它查看了100个样本和它们的正确答案")
    print("3. 每次猜错的时候，它都会把权重稍微调整一下")
    print("4. 经过50轮训练，它已经学会正确分类这些点了！")
    print()
    print("🎉 你刚刚从零开始亲手打造了一个神经网络！")
    print()
    print("🚀 试试看:")
    print("   - 调整样本数量（num_samples）：用更多或更少的例子来训练")
    print("   - 调整训练轮数（epochs）的数值，控制训练过程的长短")
    print("   - 修改学习速率（在第105行），看看会发生什么")
    print("   - 尝试不同的决策边界（修改 generate_training_data 函数）")
    print()


if __name__ == "__main__":
    main()