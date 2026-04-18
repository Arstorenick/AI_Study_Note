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
        
        参数：
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
            
            # 每 20 轮打印一次进度
            if (epoch + 1) % 20 == 0:
                avg_error = total_error / len(training_data)
                print(f"第 {epoch + 1}/{epochs} 轮 - 平均误差: {avg_error:.4f} - 权重: {self.weight:.2f}")
        
        print(f"✅ 训练完成！最终权重: {self.weight:.2f}")


def main():
    """
    主函数 —— 让我们开始教 AI 吧！
    """
    print("=" * 60)
    print("欢迎来到AI的世界！")
    print("=" * 60)
    print()
    print("今天，我们要教AI学习一个简单的数学规律：")
    print("给定 x，预测 y，满足 y = 2x 的关系")
    print()
    
    # 步骤一：准备训练数据
    # 我们希望AI学习的规律是: y = 2 * x
    print("📊 正在创建训练数据...")
    training_data = [
        (1, 2),    # 当 x = 1 时，y 应该= 2
        (2, 4),    # 当 x = 2 时，y 应该= 4
        (3, 6),    # 当 x = 3 时，y 应该= 6
        (4, 8),    # 当 x = 4 时，y 应该= 8
        (5, 10),   # 当 x = 5 时，y 应该= 10
    ]
    print(f"训练样本： {training_data}")
    print()
    
    # 步骤二: 构建并训练AI模型
    ai = SimpleAILearner()
    ai.train(training_data, epochs=100)
    print()
    
    # 步骤三: 使用新数据测试AI模型
    print("🧪 使用新数据对我们的AI进行测试……")
    print("-" * 60)
    test_inputs = [6, 7, 10, 15]
    
    for x in test_inputs:
        prediction = ai.predict(x)
        actual = 2 * x  # 正确答案
        print(f"输入值: {x:2d} | 预测结果: {prediction:6.2f} | 正确答案: {actual:6.2f} | 误差: {abs(prediction - actual):.2f}")
    
    print("-" * 60)
    print()
    
    # 解析
    print("💡 刚才发生了什么？")
    print("1. 我们向AI展示了这个数学规律（y = 2x）的样本")
    print("2. AI通过调整它的“weight”来最小化误差，从而学会了这个规律")
    print("3. 训练结束后，它就能针对新的输入预测出结果啦！")
    print()
    print("🎉 恭喜你！你刚刚训练出了你的第一个AI！")
    print()
    print("🚀 下一步：")
    print("   - 试着修改训练数据，让它学习不同的规律")
    print("   - 试着调整一下学习速率（在第29行）")
    print("   - 修改训练轮数（epochs），看看训练次数是如何影响准确率的")
    print()


if __name__ == "__main__":
    # 当你运行这个脚本时，这段代码就会执行
    main()