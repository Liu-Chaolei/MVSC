import pytest
from src.cogkit.utils import diffusion_pipeline, dtype, misc, path, random

MODEL_ID = "THUDM/CogVideoX-2b"
IMAGE_FILE = "dog.jpg"
VIDEO_FILE = "dog.mp4"


@pytest.mark.parametrize("model_id_or_path", [MODEL_ID])
# test utils.diffusion_pipeline
def test_get_pipeline_meta(model_id_or_path):
    res = diffusion_pipeline.get_pipeline_meta(model_id_or_path)
    assert res is not None


@pytest.mark.parametrize("dtypes", ["", "bfloat16", "float16"])
# test utils.dtype
def test_cast_to_torch_dtype(dtypes):
    if dtypes == "":
        with pytest.raises(ValueError) as exc_info:
            dtype.cast_to_torch_dtype(dtypes)
        assert "Unknown data type" in str(exc_info.value)
    else:
        res = dtype.cast_to_torch_dtype(dtypes)
        assert str(dtypes) in str(res)


@pytest.mark.parametrize("model_id_or_path", [MODEL_ID])
@pytest.mark.parametrize("task", ["t2i", "t2v", "i2v", "v2v"])
@pytest.mark.parametrize(
    "multimodel",
    [
        [None, None],
        [IMAGE_FILE, None],
        [None, VIDEO_FILE],
        [IMAGE_FILE, VIDEO_FILE],
    ],
)
# test utils.misc
def test_guess_generation_mode(model_id_or_path, task, multimodel):
    image_file, video_file = multimodel
    res = misc.guess_generation_mode(
        model_id_or_path,
        task,
        image_file,
        video_file,
    )
    assert task == res.value


@pytest.mark.parametrize("save_path", ["CogVideoX-2b"])
# test utils.path
def test_mkdir(save_path):
    res = path.mkdir(save_path)
    assert res is not None


@pytest.mark.parametrize("seed", [None, 42])
# test utils.random
def test_rand_generator(seed):
    res = random.rand_generator(seed)
    assert res is not None
