import streamlit as st
import torch
from PIL import Image, ImageDraw
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from datasets import load_dataset
from torchvision.ops import batched_nms

st.set_page_config(
    page_title="OWLv2 Object Detection",
    page_icon="🔍",
    layout="wide"
)

MODEL_ID = "google/owlv2-base-patch16"

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model, device

@st.cache_resource
def load_images():
    dataset = load_dataset(
        "finedet/coco2017",
        split="validation",
        streaming=True
    )
    return [
        sample["image"].convert("RGB")
        for sample in dataset.take(20)
    ]

def detect_objects(image, prompt, threshold):
    processor, model, device = load_model()

    labels = [p.strip() for p in prompt.split(",") if p.strip()]
    if not labels:
        return image.copy(), 0

    inputs = processor(
        text=[labels],
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=torch.tensor([[image.height, image.width]]),
        threshold=threshold,
        text_labels=[labels]
    )[0]

    boxes = results["boxes"].detach().cpu()
    scores = results["scores"].detach().cpu()
    class_ids = results["labels"].detach().cpu()

    if len(boxes) == 0:
        return image.copy(), 0

    keep = batched_nms(
        boxes,
        scores,
        class_ids,
        iou_threshold=0.5
    )

    output = image.copy()
    draw = ImageDraw.Draw(output)

    for i in keep.tolist():
        x1, y1, x2, y2 = boxes[i].tolist()
        score = scores[i].item()
        class_id = int(class_ids[i].item())

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )
        draw.text(
            (x1, max(0, y1 - 15)),
            f"{labels[class_id]}: {score:.2f}",
            fill="red"
        )

    return output, len(keep)


if "image_index" not in st.session_state:
    st.session_state.image_index = 0

if "result" not in st.session_state:
    st.session_state.result = None

if "result_count" not in st.session_state:
    st.session_state.result_count = 0

if "result_key" not in st.session_state:
    st.session_state.result_key = None


st.title("🔍 OWLv2 Object Detection")
st.caption("Interactive Open-Vocabulary Object Detection")
st.divider()

left, right = st.columns([1, 1.5], gap="large")

with left:
    st.subheader("Configuration")

    st.selectbox("Model Select", ["OWLv2 Base"])

    source = st.selectbox(
        "Dataset / Image Select",
        ["COCO 2017", "Upload Image"]
    )

    uploaded = None
    if source == "Upload Image":
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"]
        )

    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.15,
        step=0.05
    )

    prompt = st.text_area(
        "User Prompt",
        value="a person, a car, a chair"
    )

    run = st.button(
        "▶ Run Detection",
        type="primary",
        use_container_width=True
    )


image = None
images = []

if source == "COCO 2017":
    with st.spinner("Loading COCO images..."):
        images = load_images()
    st.session_state.image_index %= len(images)
    image = images[st.session_state.image_index]

elif uploaded is not None:
    image = Image.open(uploaded).convert("RGB")


if image is not None:
    image_key = (
        source,
        st.session_state.image_index if source == "COCO 2017"
        else uploaded.getvalue(),
        prompt,
        threshold
    )

    if run:
        try:
            with st.spinner("Running OWLv2 inference..."):
                result, count = detect_objects(
                    image, prompt, threshold
                )
            st.session_state.result = result
            st.session_state.result_count = count
            st.session_state.result_key = image_key
        except Exception as e:
            st.error(f"Detection failed: {e}")


with right:
    st.subheader("AI Predicted Image")

    if image is None:
        st.info("Please select or upload an image.")
    elif (
        st.session_state.result is not None
        and st.session_state.result_key == image_key
    ):
        st.image(
            st.session_state.result,
            caption="OWLv2 Detection Result",
            use_container_width=True
        )
        st.success(
            f"Detected objects: {st.session_state.result_count}"
        )
    else:
        st.image(
            image,
            caption="Original Image - Click Run Detection",
            use_container_width=True
        )

    prev, nxt = st.columns(2)

    with prev:
        if st.button(
            "⬅ Previous Image",
            use_container_width=True,
            disabled=(source != "COCO 2017")
        ):
            st.session_state.image_index = (
                st.session_state.image_index - 1
            ) % len(images)
            st.rerun()

    with nxt:
        if st.button(
            "Next Image ➡",
            use_container_width=True,
            disabled=(source != "COCO 2017")
        ):
            st.session_state.image_index = (
                st.session_state.image_index + 1
            ) % len(images)
            st.rerun()

    if source == "COCO 2017":
        st.caption(
            f"Image {st.session_state.image_index + 1} / {len(images)}"
        )

st.divider()
st.caption("OWLv2 | Streamlit Frontend")
