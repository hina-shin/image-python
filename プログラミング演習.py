print("画像分類")
print("Let's Pray")
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageDraw, ImageFont


# データとラベルのリスト
data = []
labels = []

# 画像フォルダのパス
emperor_penguin_path = "emperor_penguin_images"  # エンペラーペンギンの画像フォルダ
king_penguin_path = "king_penguin_images"  # キングペンギンの画像フォルダ

# エンペラーペンギン画像の読み込み
for img_file in os.listdir(emperor_penguin_path):
    img = cv2.imread(os.path.join(emperor_penguin_path, img_file))
    img = cv2.resize(img, (64, 64))  # 画像のサイズを統一する
    data.append(img)
    labels.append(0)  # エンペラーペンギンのラベルを0とする

# キングペンギン画像の読み込み
for img_file in os.listdir(king_penguin_path):
    img = cv2.imread(os.path.join(king_penguin_path, img_file))
    img = cv2.resize(img, (64, 64))
    data.append(img)
    labels.append(1)  # キングペンギンのラベルを1とする

# データのシャッフル
data = np.array(data)
labels = np.array(labels)
shuffle_indices = np.random.permutation(len(data))
data = data[shuffle_indices]
labels = labels[shuffle_indices]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#モデルの構築
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#モデルのコンパイル
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#モデルのトレーニング
model.fit(data, labels, epochs=10, batch_size=32)

# 予測関数の修正：画像を読み込み、ラベルを画像上に表示
def predict_and_show_image():

    # ファイル選択ダイアログの表示
    Tk().withdraw()  # Tkinterのウィンドウを非表示にする
    image_path = askopenfilename(title="画像を選択してください", filetypes=[("画像ファイル", "*.jpg *.jpeg *.png")])
    
    if not image_path:
        print("画像が選択されませんでした。")
        return

    print(f"選択された画像パス: {image_path}")

    image_path = os.path.normpath(image_path)
    img = cv2.imread(image_path)

    # 画像の読み込みとエラーチェック
    img = cv2.imread(image_path)
    if img is None:
        print("画像の読み込みに失敗しました。ファイルが存在するか、有効な画像形式であることを確認してください。")
        return
    
    # 画像の読み込みと前処理
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (64, 64)) / 255.0  # サイズを変更しスケーリング
    img_resized = np.expand_dims(img_resized, axis=0)

    # 予測の実行
    prediction = model.predict(img_resized)
    print(f"予測結果の確率値: {prediction[0][0]}")  # 確率値を表示
    
    label = "キングペンギン" if prediction[0][0] > 0.4 else "エンペラーペンギン"

    # 元の画像に予測結果を追加して表示
    output_img = cv2.resize(img, (512, 512))  # 表示用に大きくリサイズ

    # OpenCV画像をPillow形式に変換
    output_img_pil = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))

    # 日本語フォントの設定
    font_path = "NotoSansJP-VariableFont_wght.ttf"  # 日本語フォントのパス（例: NotoSansJP-Regular.otf）
    font = ImageFont.truetype(font_path, 32)

    # 画像に日本語を描画
    draw = ImageDraw.Draw(output_img_pil)
    draw.text((10, 10), f"予測結果: {label}", font=font, fill=(255, 0, 0))

    # Pillow画像をOpenCV形式に変換
    output_img = cv2.cvtColor(np.array(output_img_pil), cv2.COLOR_RGB2BGR)

    # 画像表示
    cv2.imshow("Prediction Result", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 予測の実行と表示
predict_and_show_image()
