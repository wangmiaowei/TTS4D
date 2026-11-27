// Site configuration: edit these links in one place.
window.TTS_CONFIG = {
  // Videos are served from a Hugging Face dataset repository (fast CDN, keeps this repository small).
  mediaBase: "https://huggingface.co/datasets/miaoweiwang/TTS4D-media/resolve/main/",
  arxiv: "",            // the paper: e.g. "https://arxiv.org/abs/XXXX.XXXXX"
  youtubeId: "",        // e.g. "dQw4w9WgXcQ"
  viewer: "http://47.102.38.61:18080/?key=425d97c5c7e770051d03"
};
// Local preview: read the encoded media from disk instead of the Hugging Face repository.
if (location.hostname === "localhost" || location.hostname === "127.0.0.1") {
  window.TTS_CONFIG.mediaBase = "/project_page_build/media/";
} else if (location.protocol === "file:") {
  window.TTS_CONFIG.mediaBase = "../../project_page_build/media/";
}
