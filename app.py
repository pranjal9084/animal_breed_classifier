# import streamlit as st
# import torch
# from PIL import Image
# from torchvision import transforms
# from enhanced_cattle_classifier import EnhancedCattleClassifier  # keep class definition in this file

# # -------------------------------
# # Load model (cached)
# # -------------------------------
# @st.cache_resource
# def load_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     checkpoint = torch.load("final_enhanced_cattle_classifier.pth", map_location=device)

#     backbone = checkpoint.get('architecture', 'resnet').replace("Enhanced", "").lower()

#     model = EnhancedCattleClassifier(num_classes=checkpoint['num_classes'], backbone=backbone).to(device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()

#     return model, device, checkpoint['class_names']

# model, device, class_names = load_model()

# # -------------------------------
# # Streamlit UI
# # -------------------------------
# st.title("🐄 Enhanced Cattle & Buffalo Breed Classifier")

# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])
#     input_tensor = preprocess(image).unsqueeze(0).to(device)

#     if st.button("Predict"):
#         with torch.no_grad():
#             outputs = model(input_tensor)
#             probs = torch.softmax(outputs, dim=1)

#             top3_probs, top3_idx = torch.topk(probs, k=3, dim=1)
#             st.success("✅ Top Predictions:")
#             for i in range(3):
#                 breed = class_names[top3_idx[0, i].item()]
#                 confidence = top3_probs[0, i].item() * 100
#                 st.write(f"{i+1}. **{breed}** — {confidence:.2f}%")




# import streamlit as st
# from inference_sdk import InferenceHTTPClient
# from PIL import Image

# # Roboflow client
# client = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key="LndSE12xFENlsDxpR418"
# )

# st.title("Roboflow Workflow Classifier")

# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     if st.button("Predict with Roboflow"):
#         # Save temp file for API
#         temp_path = "temp_image.jpg"
#         image.save(temp_path)

#         result = client.run_workflow(
#             workspace_name="project-odijn",
#             workflow_id="detect-and-classify-2",
#             images={"image": temp_path},
#             use_cache=True
#         )

#         st.json(result)  # nicely display JSON result


# import streamlit as st
# import torch
# import timm
# import numpy as np
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2


# # -- Parameters --
# NUM_CLASSES = 42    # set to number of animal breeds
# IMG_SIZE = 224
# MODEL_PATH = "C:\\Users\\baniy\\Desktop\\project_sih\\output\\new_model.pth"


# # ----- Label mapping -----
# classes = ['Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Surti buffalo', 'Tharparkar', 'Toda', 'Umblachery', 'Vechur']  # fill this with your class names
# idx2label = {i: c for i, c in enumerate(classes)}

# # ----- Preprocessing pipeline -----
# val_transform = A.Compose([
#     A.Resize(IMG_SIZE, IMG_SIZE),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2(),
# ])

# # ----- Model definition -----
# def create_model(num_classes, model_name='tf_efficientnet_b0_ns', drop_rate=0.4):
#     model = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='avg')
#     in_features = model.num_features
#     head = torch.nn.Sequential(
#         torch.nn.Linear(in_features, 512),
#         torch.nn.ReLU(),
#         torch.nn.Dropout(p=drop_rate),
#         torch.nn.Linear(512, num_classes)
#     )
#     model.classifier = head
#     return model

# # ----- Load model -----
# @st.cache_resource
# def load_model():
#     model = create_model(len(classes))
#     model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
#     model.eval()
#     return model

# model = load_model()

# # ----- Prediction function -----
# def predict_image(image, model):
#     img = np.array(image.convert("RGB"))
#     img = val_transform(image=img)['image']
#     img = img.unsqueeze(0)
#     with torch.no_grad():
#         logits = model(img)
#         pred = torch.argmax(logits, dim=1).item()
#         label = idx2label[pred]
#     return label

# # ----- Streamlit UI -----
# st.title("Animal Breed Classification Demo")
# st.write("Upload an animal image to get the predicted breed.")

# img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

# if img_file is not None:
#     image = Image.open(img_file)
#     st.image(image, caption="Uploaded Image", width=256)
#     if st.button("Classify"):
#         breed = predict_image(image, model)
#         st.success(f"Predicted breed: **{breed}**")



# with excel sheet

import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

# --- Load breed info Excel ---
BREED_INFO_PATH = r"C:\Users\baniy\Desktop\project_sih\output\breed_morphology.csv"
breed_info_df = pd.read_csv(BREED_INFO_PATH)  # use read_csv for CSV files
breed_info_df['Breed'] = breed_info_df['Breed'].str.lower()


# --- Constants ---
NUM_CLASSES = 42
IMG_SIZE = 224
MODEL_PATH = r"C:\Users\baniy\Desktop\project_sih\output\new_model.pth"

# --- Classes and mappings ---
classes = ['Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Surti buffalo', 'Tharparkar', 'Toda', 'Umblachery', 'Vechur']
idx2label = {i: c for i, c in enumerate(classes)}

# --- Preprocessing ---
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --- Model ---
def create_model(num_classes, model_name='tf_efficientnet_b0_ns', drop_rate=0.4):
    model = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='avg')
    in_features = model.num_features
    head = torch.nn.Sequential(
        torch.nn.Linear(in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=drop_rate),
        torch.nn.Linear(512, num_classes)
    )
    model.classifier = head
    return model

@st.cache_resource
def load_model():
    model = create_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# --- Prediction function ---
def predict_image(image, model):
    img = np.array(image.convert("RGB"))
    img = val_transform(image=img)['image']
    img = img.unsqueeze(0)
    with torch.no_grad():
        logits = model(img)
        pred = torch.argmax(logits, dim=1).item()
        label = idx2label[pred]
    return label

# --- Streamlit UI ---

st.title("Animal Breed Classification Demo")
st.write("Upload an animal image to get the predicted breed, height, and weight details.")

img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption="Uploaded Image", width=256)
    if st.button("Classify"):
        breed = predict_image(image, model).lower()
        st.success(f"Predicted breed: **{breed}**")

        info = breed_info_df[breed_info_df['Breed'] == breed]
        if not info.empty:
            height = info['Height (avg. cm.)'].values[0]
            weight = info['Heart girth (avg. cm.)'].values[0]
            horn = info['Horn shape and size'].values[0]
            color = info['Colour'].values[0]
            st.write(f"Height(avg) :  {height} cm")
            st.write(f"Heart girth(avg cm.):  {weight} kg")
            st.write(f"Horn shape and size :  {horn} ")
            st.write(f"Colour :  {color}" )
        else:
            st.warning("No height/weight info found for this breed.")
