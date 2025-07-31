import os
import numpy
import matplotlib.pyplot as plt
from PIL import Image
from neural_network_class import neuralNetwork

#定义文件路径和模型架构
INPUT_NODES = 784
HIDDEN_NODES = 200
OUTPUT_NODES = 10
LEARNING_RATE = 0.1

MODEL_SAVE_PATH = os.path.join('saved_model', 'trained_weights.npz')
TEST_IMAGES_DIR = 'test_images'

def predict_images(model):
    if not os.path.exists(TEST_IMAGES_DIR) or not os.listdir(TEST_IMAGES_DIR):
        print(f"文件夹 '{TEST_IMAGES_DIR}' 不存在或为空。")
        return

    print(f"\n开始识别 '{TEST_IMAGES_DIR}' 文件夹中的图片（将文件命名为如'8.png' 的形式，程序会自动进行验证）")
    
    for image_file in os.listdir(TEST_IMAGES_DIR):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(TEST_IMAGES_DIR, image_file)
            print(f"\n-------------------------------------------------")
            print(f"图片{image_file}：")
            
            #验证预测结果
            correct_label = None
            try:
                # 从文件名的第一个字符获取正确答案
                correct_label = int(image_file[0])
            except (ValueError, IndexError):
                # 如果文件名不以数字开头，则跳过验证
                print("文件名不以数字开头，无法进行自动验证。")
            
            try:
                img = Image.open(image_path).convert('L')
                img = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                img_data = numpy.asarray(img)
                img_data_inverted = 255.0 - img_data
                scaled_input = (img_data_inverted / 255.0 * 0.99) + 0.01
                image_vector = scaled_input.flatten()
                
                outputs = model.query(image_vector)
                prediction = numpy.argmax(outputs)

                #显示模型置信度
                print("模型置信度分布:")
                # .flatten() 确保 outputs 是一个一维数组
                for i, prob in enumerate(outputs.flatten()):
                    print(f"  - 数字 {i}: {prob*100:.2f}%")

                print(f"\n模型的最终预测是: {prediction}")

                if correct_label is not None:
                    print(f"文件名提供的正确答案是: {correct_label}")
                    if prediction == correct_label:
                        print("结果: 正确")
                    else:
                        print("结果: 错误")
                
                # 显示结果
                plt.imshow(img_data_inverted, cmap='gray')
                plt.title(f"Model Prediction: {prediction}")
                plt.show()

            except Exception as e:
                print(f"处理图片 '{image_file}' 时发生错误: {e}")

#主程序入口
if __name__ == "__main__":
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"错误: 找不到已训练的模型 '{MODEL_SAVE_PATH}'，请先运行 'train.py'")
    else:
        print("初始化神经网络架构...")
        nn = neuralNetwork(inputnodes=INPUT_NODES, 
                           hiddennodes=HIDDEN_NODES, 
                           outputnodes=OUTPUT_NODES, 
                           learningrate=LEARNING_RATE)
        
        nn.load_weights(MODEL_SAVE_PATH)
        predict_images(nn)