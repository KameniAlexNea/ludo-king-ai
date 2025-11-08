import base64
import io


class Utils:
    """Utility functions for the application."""

    @staticmethod
    def img_to_data_uri(pil_img):
        """Return an inline data URI for the PIL image to avoid Gradio temp file folders."""
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return (
            "<div style='display:flex;justify-content:center;'>"
            f"<img src='data:image/png;base64,{b64}' "
            "style='image-rendering:pixelated;max-width:800px;width:100%;height:auto;border-radius:10px;box-shadow:0 4px 8px rgba(0,0,0,0.1);' />"
            "</div>"
        )