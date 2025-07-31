import os
import numpy
import pandas
from neural_network_class import neuralNetwork

# --- 1. 定义文件路径和模型超参数 ---
TRAIN_DATA_FILE = os.path.join('data', 'mnist_train.csv')
TEST_DATA_FILE = os.path.join('data', 'mnist_test.csv')
MODEL_SAVE_PATH = os.path.join('saved_model', 'trained_weights.npz') # 模型保存路径

INPUT_NODES = 784
HIDDEN_NODES = 200
OUTPUT_NODES = 10
LEARNING_RATE = 0.1
EPOCHS = 5

def train_model(model):
    print("开始加载训练数据...")
    try:
        data = pandas.read_csv(TRAIN_DATA_FILE, header=None).values
    except FileNotFoundError:
        print(f"错误: 找不到训练文件 '{TRAIN_DATA_FILE}'")
        return False
    
    print("数据加载完成，开始训练模型...")

    for e in range(EPOCHS):
        print(f"正在进行第 {e + 1}/{EPOCHS} 轮训练")
        for record in data:
            inputs = (record[1:] / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(OUTPUT_NODES) + 0.01
            targets[int(record[0])] = 0.99
            model.train(inputs, targets)
            
    print("\n模型训练完成.")
    return True

def evaluate_accuracy(model):
    print("\n开始在测试集上评估模型准确率...")
    try:
        test_data = pandas.read_csv(TEST_DATA_FILE, header=None).values
    except FileNotFoundError:
        print(f"Error: 找不到测试文件 '{TEST_DATA_FILE}'")
        return

    scorecard = []
    for record in test_data:
        correct_label = int(record[0])
        inputs = (record[1:] / 255.0 * 0.99) + 0.01
        outputs = model.query(inputs)
        predicted_label = numpy.argmax(outputs)
        if (predicted_label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
    
    accuracy = sum(scorecard) / len(scorecard)
    print(f"模型准确率: {accuracy:.4f} ({accuracy:.2%})")

#主程序
if __name__ == "__main__":
    print("初始化新的神经网络模型...")
    nn = neuralNetwork(inputnodes=INPUT_NODES, 
                       hiddennodes=HIDDEN_NODES, 
                       outputnodes=OUTPUT_NODES, 
                       learningrate=LEARNING_RATE)

    if train_model(nn):
        evaluate_accuracy(nn)
        # 训练和评估完成后保存权重
        nn.save_weights(MODEL_SAVE_PATH)