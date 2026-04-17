"""
Hello AI World - 你的第一个AI程序
=======================================

这是一个简单的模式识别示例，展示了人工智能的核心概念：
·从数据中学习
·进行预测
·理解模式

该程序的功能：
·学习一个简单的数学规则（y = 2x）
·利用该模式进行预测
·不需要复杂的库———只需纯 Python！

非常适合在深入神经网络之前理解人工智能的基础知识。
"""

import random

class SimpleAILearner:
    """
    一个非常简单的AI，专门学习线性关系。
    这展示了人工智能的核心概念：从数据中学习。
    """
    
    def __init__(self):
        # "weight" 就是我们要让AI去学习的东西
        # 它从一个随机的猜测开始
        self.weight = random.uniform(0, 5)
        self.learning_rate = 0.01  # AI的学习速率
        
    def predict(self, x):
        """
        根据我们已学习到的内容进行预测。

        参数:
            x: 输入值
    
        返回:
            预测的输出结果
        """

        return self.weight * x
    
    def train(self, training_data, epochs=100):
        """
        训练 AI 掌握数据背后的规律。
        
        Args:
            training_data: 包含 (输入, 输出) 配对的列表
            epochs: 训练轮数
        """
        print("🎓 训练开始了...")
        print(f"weight的初始值: {self.weight:.2f}")
        
        for epoch in range(epochs):
            total_error = 0
            
            # 从每个样本中学习
            for x, y_actual in training_data:
                # 做出预测
                y_predicted = self.predict(x)
                
                # 计算误差 (自己跟正确答案之间差了多远)
                error = y_actual - y_predicted
                total_error += abs(error)
                
                # 调整权重，降低误差 (这才是真正的学习！)
                self.weight += self.learning_rate * error * x
            
            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                avg_error = total_error / len(training_data)
                print(f"Epoch {epoch + 1}/{epochs} - Average error: {avg_error:.4f} - Weight: {self.weight:.2f}")
        
        print(f"✅ Training complete! Final weight: {self.weight:.2f}")


def main():
    """
    Main function - Let's teach our AI!
    """
    print("=" * 60)
    print("Welcome to Hello AI World!")
    print("=" * 60)
    print()
    print("Today, we'll teach an AI to learn a simple pattern:")
    print("Given x, predict y where y = 2x")
    print()
    
    # Step 1: Create training data
    # The pattern we want the AI to learn: y = 2 * x
    print("📊 Creating training data...")
    training_data = [
        (1, 2),    # When x=1, y should be 2
        (2, 4),    # When x=2, y should be 4
        (3, 6),    # When x=3, y should be 6
        (4, 8),    # When x=4, y should be 8
        (5, 10),   # When x=5, y should be 10
    ]
    print(f"Training examples: {training_data}")
    print()
    
    # Step 2: Create and train our AI
    ai = SimpleAILearner()
    ai.train(training_data, epochs=100)
    print()
    
    # Step 3: Test our AI with new data
    print("🧪 Testing our AI with new inputs...")
    print("-" * 60)
    test_inputs = [6, 7, 10, 15]
    
    for x in test_inputs:
        prediction = ai.predict(x)
        actual = 2 * x  # The true answer
        print(f"Input: {x:2d} | Prediction: {prediction:6.2f} | Actual: {actual:6.2f} | Difference: {abs(prediction - actual):.2f}")
    
    print("-" * 60)
    print()
    
    # Explanation
    print("💡 What just happened?")
    print("1. We gave the AI examples of the pattern (y = 2x)")
    print("2. The AI learned by adjusting its 'weight' to minimize errors")
    print("3. After training, it can predict outputs for new inputs!")
    print()
    print("🎉 Congratulations! You just trained your first AI!")
    print()
    print("🚀 Next steps:")
    print("   - Try changing the training data to learn different patterns")
    print("   - Experiment with the learning_rate (line 29)")
    print("   - Modify epochs to see how training time affects accuracy")
    print()


if __name__ == "__main__":
    # This runs when you execute the script
    main()