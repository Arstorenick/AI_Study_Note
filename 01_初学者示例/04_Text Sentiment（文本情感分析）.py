"""
简单的文本情感分析
================================

本示例展示了如何分析文本的情感（情绪）。
这是一个简化版本，旨在不依赖复杂库的情况下教授自然语言处理的概念。

你将会学到:
- 文本预处理(Text preprocessing)(清洗和准备文本)
- 特征提取(Feature extraction)(将单词转换为数字)
- 情感分类(Sentiment classification)(正面与负面)

应用案例：判断一条电影评论是好评还是差评。
"""

import re
from collections import Counter

class SimpleSentimentAnalyzer:
    """
    一个基础的、能从带标签的样本中进行学习的情感分析器。
    
    工作原理：
    1. 学习哪些词汇在正面或负面文本中出现得更频繁
    2. 计算每个词汇的“情感得分”
    3. 利用这些得分来预测新文本的情感倾向
    """
    
    def __init__(self):
        # 存储词汇得分（正面词汇获得正分）
        self.word_scores = {}
        # 记录训练状态
        self.is_trained = False
        
    def preprocess_text(self, text):
        """
        清洗并准备用于分析的文本。
        
        步骤:
        1. 转换为小写
        2. 去除标点符号
        3. 分割成单词
        
        参数:
            text: 原始文本字符串
            
        返回:
            清洗后的单词列表
        """
        # 转换为小写
        text = text.lower()
        
        # 去除标点符号和特殊字符
        text = re.sub(r'[^a-z\s]', '', text)
        
        # 分割成单词
        words = text.split()
        
        # 去除极短的单词（比如 "a", "i"）
        words = [w for w in words if len(w) > 2]
        
        return words
    
    def train(self, training_data):
        """
         从带标签的样本中学习情感模式。
        
        Args:
            training_data: 由 (文本text, 情感sentiment) 元组组成的列表
                          其中情感标签为 'positive'（正面）或 'negative'（负面）
        """
        print("🎓 正在训练情感分析器...")
        
        # 统计正面和负面文本中的词频
        positive_words = Counter()
        negative_words = Counter()
        
        for text, sentiment in training_data:
            words = self.preprocess_text(text)
            
            if sentiment == 'positive':
                positive_words.update(words)
            else:
                negative_words.update(words)
        
        # 计算每个单词的情感得分
        # 得分 > 0 表示偏正面，< 0 表示偏负面
        all_words = set(positive_words.keys()) | set(negative_words.keys())
        
        for word in all_words:
            pos_count = positive_words[word]
            neg_count = negative_words[word]
            
            # 计算得分：计算出现次数的差值
            # 加上平滑项 (+1) 以避免除以零
            total = pos_count + neg_count
            self.word_scores[word] = (pos_count - neg_count) / (total + 1)
        
        self.is_trained = True
        
        # 展示一些学习到的单词
        print(f"✅ 已学习 {len(self.word_scores)} 个单词的情感倾向")
        print("\n📊 最正面的词汇:")
        sorted_words = sorted(self.word_scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:5]:
            print(f"   '{word}': {score:+.3f}")
        
        print("\n📊 最负面的词汇:")
        for word, score in sorted_words[-5:]:
            print(f"   '{word}': {score:+.3f}")
    
    def analyze(self, text):
        """
        预测新文本的情感倾向。
        
        Args:
            text: 待分析的文本
            
        Returns:
            包含 (情感倾向, 置信度, 分数) 的元组(sentiment, confidence, score)
        """
        if not self.is_trained:
            raise Exception("请先对分析器进行训练！")
        
        # 文本预处理(Preprocess text)
        words = self.preprocess_text(text)
        
        # 计算情感总分
        total_score = 0
        word_count = 0
        
        for word in words:
            if word in self.word_scores:
                total_score += self.word_scores[word]
                word_count += 1
        
        # 平均得分
        if word_count > 0:
            avg_score = total_score / word_count
        else:
            avg_score = 0
        
        # 判定情感倾向与置信度
        sentiment = "positive" if avg_score > 0 else "negative"
        confidence = min(abs(avg_score) * 100, 100)  # 转换为百分比
        
        return sentiment, confidence, avg_score


def create_training_data():
    """
    创建示例训练数据（带标签的影评）。
    
    在真实的应用中，你可是会有成千上万个例子呢！
    
    返回值:
        包含 (评论文本, 情感倾向) 的元组列表
    """
    return [
        # 正面评论
        ("This movie was absolutely amazing and wonderful! I loved every minute.", "positive"),
        ("Brilliant performance! The acting was superb and the story captivating.", "positive"),
        ("Fantastic film! Highly recommend to everyone. Best movie of the year!", "positive"),
        ("Loved it! Great storytelling and beautiful cinematography.", "positive"),
        ("Excellent movie with outstanding performances. A must watch!", "positive"),
        ("Amazing! This film exceeded all my expectations. Truly remarkable.", "positive"),
        ("Wonderful experience! The plot was engaging and entertaining.", "positive"),
        ("Superb direction and acting! One of the best films I've seen.", "positive"),
        
        # 负面评论
        ("Terrible movie. Waste of time and money. Very disappointed.", "negative"),
        ("Awful film! Poor acting and boring story. Would not recommend.", "negative"),
        ("Horrible! The worst movie I have ever seen. Extremely disappointing.", "negative"),
        ("Bad movie with terrible plot. Boring and predictable.", "negative"),
        ("Disappointing film. Poor execution and weak performances.", "negative"),
        ("Worst movie ever! Horrible acting and stupid storyline.", "negative"),
        ("Terrible experience. Boring and poorly made. Don't waste your time.", "negative"),
        ("Awful! Poor quality and uninteresting. Complete waste of time.", "negative"),
    ]


def main():
    """
    主函数 —— 让我们来搞点情感分析吧！
    """
    print("=" * 70)
    print("简单的文本情感分析")
    print("=" * 70)
    print("\n📚 任务：学会识别影评是正面还是负面")
    print()
    
    # 步骤一: 创建训练数据
    training_data = create_training_data()
    print(f"📊 训练数据: {len(training_data)} 条影评")
    print()
    
    # 步骤二: 创建并训练分析器
    analyzer = SimpleSentimentAnalyzer()
    analyzer.train(training_data)
    print()
    
    # 步骤三: 在新影评上进行测试
    print("🧪 正在对新影评进行测试:")
    print("=" * 70)
    
    test_reviews = [
        "This movie was fantastic! I really enjoyed it.",
        "Boring and terrible. Not worth watching.",
        "Amazing cinematography and wonderful acting!",
        "The worst film I've seen this year. Awful.",
        "Pretty good movie with some great moments.",
        "Disappointing and poorly directed.",
    ]
    
    for i, review in enumerate(test_reviews, 1):
        sentiment, confidence, score = analyzer.analyze(review)
        
        # 可视化标识
        indicator = "😊" if sentiment == "positive" else "😞"
        
        print(f"\n评论 {i}:")
        print(f"  内容: \"{review}\"")
        print(f"  {indicator} 情感倾向: {sentiment.upper()}")
        print(f"  📊 置信度: {confidence:.1f}%")
        print(f"  📈 得分: {score:+.3f}")
    
    print("\n" + "=" * 70)
    
    # Interactive mode
    print("\n💬 亲自试一试！输入你自己的评论（或者输入 'quit' 退出）:")
    print("-" * 70)
    
    while True:
        user_input = input("\n请输入你的评论: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        try:
            sentiment, confidence, score = analyzer.analyze(user_input)
            indicator = "😊" if sentiment == "positive" else "😞"
            
            print(f"\n{indicator} 情感倾向: {sentiment.upper()}")
            print(f"📊 置信度: {confidence:.1f}%")
            print(f"📈 得分: {score:+.3f}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Explanation
    print("\n💡 刚才发生了什么？")
    print("1. 分析器从示例评论中学习到了词汇模式")
    print("2. 它计算了单词的“情感得分”(sentiment scores)")
    print("3. 对于新文本，它会综合单词得分来预测情感")
    print()
    print("🎉 恭喜你，刚刚做出了一个情感分析工具！")
    print()
    print("🚀 接下来的步骤:")
    print("   - 添加更多训练样本来提高准确率")
    print("   - 尝试分析推文、商品评论或留言")
    print("   - 在 https://github.com/microsoft/AI-For-Beginners/tree/main/translations/zh-CN/lessons/5-NLP 中探索更高级的 NLP 技术")
    print()


if __name__ == "__main__":
    main()