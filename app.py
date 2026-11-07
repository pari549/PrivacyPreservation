"""Streamlit demonstration UI for PrivacyGuard image transformation services."""
from __future__ import annotations

import hashlib
import uuid

import streamlit as st

from src.application.recoverable_service import RecoverableImageService
from src.application.ui_helpers import (
    RecoverableConfigurationError,
    UploadValidationError,
    call_non_recoverable_transform,
    call_recoverable_recovery,
    call_recoverable_transform,
    inspect_image_upload,
    recoverable_service_configuration,
    safe_service_error,
)
from src.non_recoverable.baseline_transformations import (
    MAX_QUANTIZATION_LEVELS,
    MAX_SMOOTHING_STRENGTH,
    MIN_QUANTIZATION_LEVELS,
)


st.set_page_config(
    page_title="Privacy Preservation System",
    page_icon="🔒",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _recoverable_service(storage_root: str, key_path: str) -> RecoverableImageService:
    """Share the trusted in-process run registry across Streamlit reruns."""
    return RecoverableImageService(storage_root, key_path=key_path)


def _format_metric(value: object, suffix: str = "") -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.6g}{suffix}"
    except (TypeError, ValueError):
        return "Unavailable"


def _upload_info(uploaded, mode: str):
    if uploaded is None:
        for suffix in ("result", "recovered", "job_id", "upload_digest"):
            st.session_state.pop(f"{mode}_{suffix}" if suffix != "upload_digest" else f"_{mode}_upload_digest", None)
        return None, None, None
    content = uploaded.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    state_digest_key = f"_{mode}_upload_digest"
    if st.session_state.get(state_digest_key) != digest:
        st.session_state[state_digest_key] = digest
        st.session_state.pop(f"{mode}_result", None)
        st.session_state.pop(f"{mode}_recovered", None)
        st.session_state.pop(f"{mode}_job_id", None)
    try:
        info = inspect_image_upload(content, uploaded.name)
    except UploadValidationError as error:
        st.error(str(error))
        return content, None, None
    st.image(info.preview_png, caption="Uploaded image preview", use_container_width=True)
    st.caption(
        f"{info.width} × {info.height} pixels · source mode {info.color_mode} · {info.image_format}"
    )
    return content, info, digest


def _render_recoverable() -> None:
    st.title("Recoverable Privacy")
    st.write(
        "Apply the authenticated recoverable transformation. This prototype accepts only "
        "decoded **28 × 28 RGB** images and does not resize uploads."
    )
    st.info(
        "Recovery authenticates the stored bundle and reconstructs the decoded pixels. "
        "HMAC authenticates data; it does not encrypt stored files. The run registry is "
        "process-local, so runs cannot be recovered after an application restart."
    )
    uploaded = st.file_uploader(
        "Choose a PNG, JPG, or JPEG image",
        type=["png", "jpg", "jpeg"],
        key="recoverable_upload",
    )
    content, info, _ = _upload_info(uploaded, "recoverable")
    eligible = False
    if info is not None:
        if info.is_grayscale:
            st.error("Recoverable mode requires RGB; grayscale images are not supported.")
        elif (info.height, info.width) != (28, 28):
            st.error("Recoverable mode accepts only 28 × 28 RGB images. Images are not resized.")
        else:
            eligible = True
            st.success("Upload is valid for recoverable mode.")

    if st.button("Transform image", disabled=not eligible, key="recoverable_transform"):
        try:
            storage_root, key_path = recoverable_service_configuration()
            service = _recoverable_service(str(storage_root), str(key_path))
            with st.spinner("Creating and authenticating an isolated recovery run…"):
                result = call_recoverable_transform(service, content, info)
            st.session_state["recoverable_result"] = result
            st.session_state.pop("recoverable_recovered", None)
            st.success("Transformation completed. Keep the run ID available for this session.")
        except RecoverableConfigurationError as error:
            st.error(str(error))
        except Exception as error:
            st.error(safe_service_error(error, mode="recoverable"))

    result = st.session_state.get("recoverable_result")
    if result is not None:
        st.markdown("#### Transformed image")
        left, right = st.columns(2)
        with left:
            st.image(info.preview_png if info else content, caption="Original", use_container_width=True)
        with right:
            st.image(result.transformed_image_bytes, caption="Transformed", use_container_width=True)
        st.code(result.run_id, language=None)
        st.caption("Run ID (service-issued; valid only while this server process retains its registry).")

        authorized = st.checkbox(
            "I am authorized to recover this image and have access to this run.",
            key="recoverable_authorized",
        )
        st.caption("This local prototype does not provide accounts or identity-based access control.")
        if st.button("Recover image", disabled=not authorized, key="recoverable_recover"):
            try:
                storage_root, key_path = recoverable_service_configuration()
                service = _recoverable_service(str(storage_root), str(key_path))
                with st.spinner("Authenticating artifacts and verifying exact recovery…"):
                    recovered = call_recoverable_recovery(service, result.run_id)
                st.session_state["recoverable_recovered"] = recovered
            except Exception as error:
                st.error(
                    "This run is unavailable or recovery verification failed. It may have been "
                    "created before a server restart. "
                    + safe_service_error(error, mode="recoverable")
                )
        recovered = st.session_state.get("recoverable_recovered")
        if recovered is not None:
            st.image(recovered.recovered_image_bytes, caption="Recovered image", use_container_width=True)
            if recovered.pixel_perfect:
                st.success("Recovery succeeded: the service verified exact pixel equality.")
            else:
                st.error("Recovery did not pass exact pixel equality verification.")


def _render_non_recoverable() -> None:
    st.title("Non-Recoverable Privacy")
    st.write(
        "Apply lossy block-level quantization and Gaussian smoothing to supported RGB or "
        "grayscale images. **No recovery operation is provided.**"
    )
    st.warning(
        "Global settings are applied uniformly to every block through the adaptive interface. "
        "Image features are descriptive; they do not select settings or demonstrate privacy effectiveness."
    )
    uploaded = st.file_uploader(
        "Choose a PNG, JPG, or JPEG image",
        type=["png", "jpg", "jpeg"],
        key="nonrecoverable_upload",
    )
    content, info, _ = _upload_info(uploaded, "nonrecoverable")
    if info is None:
        return

    with st.sidebar:
        st.subheader("Transformation settings")
        quantization_levels = st.slider(
            "Quantization levels",
            min_value=MIN_QUANTIZATION_LEVELS,
            max_value=MAX_QUANTIZATION_LEVELS,
            value=256,
            step=1,
            help="Number of evenly spaced quantization values, from 2 to 65,536.",
        )
        smoothing_strength = st.slider(
            "Gaussian smoothing strength (sigma)",
            min_value=0.0,
            max_value=MAX_SMOOTHING_STRENGTH,
            value=1.0,
            step=0.1,
            help="Gaussian kernel sigma in pixels. Zero leaves the smoothing step unchanged.",
        )
    grayscale = info.is_grayscale
    if st.button("Transform image", key="nonrecoverable_transform"):
        try:
            with st.spinner("Extracting block features and transforming the image…"):
                result = call_non_recoverable_transform(
                    content,
                    quantization_levels=quantization_levels,
                    smoothing_strength=smoothing_strength,
                    grayscale=grayscale,
                )
            job_id = uuid.uuid4().hex
            st.session_state["nonrecoverable_result"] = result
            st.session_state["nonrecoverable_job_id"] = job_id
            st.success("Non-recoverable transformation completed.")
        except Exception as error:
            st.error(safe_service_error(error, mode="non_recoverable"))

    result = st.session_state.get("nonrecoverable_result")
    if result is None:
        return
    st.caption(f"Job ID: {st.session_state.get('nonrecoverable_job_id', 'unavailable')}")
    before, after = st.columns(2)
    with before:
        st.image(info.preview_png, caption="Original (orientation-corrected preview)", use_container_width=True)
    with after:
        st.image(result.transformed_image_bytes, caption="Transformed", use_container_width=True)
    st.subheader("Utility metrics")
    metrics = st.columns(4)
    metrics[0].metric("MAE", _format_metric(result.metrics.get("mae")))
    metrics[1].metric("PSNR", _format_metric(result.metrics.get("psnr_db"), " dB"))
    metrics[2].metric("SSIM", _format_metric(result.metrics.get("ssim")))
    metrics[3].metric("Processing time", _format_metric(result.processing_time_seconds, " s"))
    st.caption(
        f"Applied uniformly to each block: {quantization_levels} quantization levels, "
        f"Gaussian sigma {smoothing_strength:g}. Order: quantization, then smoothing."
    )
    st.caption(
        f"{result.dimensions[1]} × {result.dimensions[0]} pixels · {result.channels} channel(s) · "
        f"{result.feature_summary['block_count']} blocks"
    )
    st.download_button(
        "Download transformed PNG",
        data=result.transformed_image_bytes,
        file_name="privacyguard-transformed.png",
        mime="image/png",
        key="nonrecoverable_download",
    )
    if result.warnings:
        with st.expander("Limitations and processing notes"):
            for warning in result.warnings:
                st.write(f"• {warning}")
    with st.expander("Preprocessing details"):
        safe_fields = {
            key: value
            for key, value in result.preprocessing_metadata.items()
            if key not in {"encoded_width", "encoded_height"}
        }
        st.json(safe_fields)


def main() -> None:
    st.title("Privacy Preservation System")
    st.write("Transform images using recoverable and non-recoverable privacy modes.")
    with st.sidebar:
        st.header("Choose a mode")
        mode = st.radio(
            "Image transformation mode",
            ("Recoverable Privacy", "Non-Recoverable Privacy"),
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Research prototype; results do not establish universal privacy protection.")
    if mode == "Recoverable Privacy":
        _render_recoverable()
    else:
        _render_non_recoverable()


if __name__ == "__main__":
    main()
