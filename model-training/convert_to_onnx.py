"""
AASIST PyTorch → ONNX 변환 스크립트

기존 deep-fake-audio-detection 프로젝트의 AASIST 모델을
ONNX 형식으로 변환하여 Android (ONNX Runtime Mobile)에서 실행 가능하게 한다.

Usage:
    python convert_to_onnx.py --model-path <pth_path> --output <onnx_path> [--verify]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 기존 AASIST 모델 코드 경로 추가
AASIST_CODE_DIR = Path(__file__).resolve().parent.parent.parent / "deep-fake-audio-detection" / "code" / "2_aasist_rawboost"
sys.path.insert(0, str(AASIST_CODE_DIR))

from models.AASIST import Model


class AASISTInference(nn.Module):
    """추론 전용 AASIST 래퍼.

    - DANN (ReverseLayerF, domain_classifier) 제거
    - forward()에서 softmax 적용된 [real_prob, fake_prob] 만 반환
    - SincConv의 band_pass 필터를 고정 버퍼로 등록 (ONNX 호환)
    """

    def __init__(self, original_model: Model):
        super().__init__()
        self.d_args = original_model.d_args

        # SincConv: band_pass 필터를 고정 파라미터로 변환
        self.conv_time_filters = nn.Parameter(
            original_model.conv_time.band_pass.unsqueeze(1), requires_grad=False
        )
        self.conv_time_stride = original_model.conv_time.stride
        self.conv_time_padding = original_model.conv_time.padding
        self.conv_time_dilation = original_model.conv_time.dilation

        # 나머지 레이어는 그대로 복사
        self.first_bn = original_model.first_bn
        self.selu = nn.SELU(inplace=False)  # inplace=False for ONNX
        self.drop = nn.Dropout(0.5, inplace=False)
        self.drop_way = nn.Dropout(0.2, inplace=False)

        self.encoder = original_model.encoder

        self.pos_S = original_model.pos_S
        self.master1 = original_model.master1
        self.master2 = original_model.master2

        self.GAT_layer_S = original_model.GAT_layer_S
        self.GAT_layer_T = original_model.GAT_layer_T

        self.HtrgGAT_layer_ST11 = original_model.HtrgGAT_layer_ST11
        self.HtrgGAT_layer_ST12 = original_model.HtrgGAT_layer_ST12
        self.HtrgGAT_layer_ST21 = original_model.HtrgGAT_layer_ST21
        self.HtrgGAT_layer_ST22 = original_model.HtrgGAT_layer_ST22

        self.pool_S = original_model.pool_S
        self.pool_T = original_model.pool_T
        self.pool_hS1 = original_model.pool_hS1
        self.pool_hT1 = original_model.pool_hT1
        self.pool_hS2 = original_model.pool_hS2
        self.pool_hT2 = original_model.pool_hT2

        self.out_layer = original_model.out_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 80000) raw audio waveform, 16kHz
        Returns:
            (batch, 2) softmax probabilities [real_prob, fake_prob]
        """
        # SincConv 대체: pre-computed 필터로 Conv1d
        x = x.unsqueeze(1)  # (batch, 1, 80000)
        x = torch.nn.functional.conv1d(
            x,
            self.conv_time_filters,
            stride=self.conv_time_stride,
            padding=self.conv_time_padding,
            dilation=self.conv_time_dilation,
        )

        x = x.unsqueeze(1)  # (batch, 1, filts, time)
        x = torch.nn.functional.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # Encoder
        e = self.encoder(x)

        # Spectral GAT
        e_S, _ = torch.max(torch.abs(e), dim=3)
        e_S = e_S.transpose(1, 2) + self.pos_S

        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        # Temporal GAT
        e_T, _ = torch.max(torch.abs(e), dim=2)
        e_T = e_T.transpose(1, 2)

        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # Learnable master nodes
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # Inference path 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1
        )
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1
        )
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # Inference path 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2
        )
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2
        )
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        # Aggregate (dropout disabled in .eval() mode)
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1
        )
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        # Softmax 적용
        output = torch.softmax(output, dim=1)
        return output


AASIST_L_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 80000,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def infer_model_config(model_path: str) -> dict:
    """checkpoint의 shape에서 AASIST vs AASIST-L을 자동 판별."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = checkpoint.get("model_state_dict", checkpoint)
    gat_dim = sd["GAT_layer_S.att_weight"].shape[0]
    if gat_dim == 24:
        print(f"  Detected AASIST-L (gat_dim={gat_dim})")
        return AASIST_L_CONFIG
    print(f"  Detected AASIST (gat_dim={gat_dim}), using AASIST.conf")
    return None  # use default config file


def load_aasist_model(model_path: str, config_path: str = None) -> Model:
    """기존 AASIST 모델 로드."""
    # 자동 config 판별
    auto_config = infer_model_config(model_path)

    if auto_config is not None:
        model_config = auto_config
    elif config_path is not None:
        with open(config_path, "r") as f:
            model_config = json.load(f)["model_config"]
    else:
        config_path = str(AASIST_CODE_DIR / "config" / "AASIST.conf")
        with open(config_path, "r") as f:
            model_config = json.load(f)["model_config"]

    model = Model(model_config)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model


def convert_to_onnx(model_path: str, output_path: str, config_path: str = None):
    """PyTorch AASIST → ONNX 변환."""
    print(f"Loading model from {model_path}...")
    original_model = load_aasist_model(model_path, config_path)

    print("Creating inference wrapper...")
    inference_model = AASISTInference(original_model)
    inference_model.eval()

    # 더미 입력 (batch=1, 80000 samples = 5초 @ 16kHz)
    dummy_input = torch.randn(1, 80000)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        inference_model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["audio"],
        output_names=["probabilities"],
        dynamic_axes=None,  # 고정 배치 크기 (1)
    )
    print(f"ONNX model saved to {output_path}")

    # 파일 크기 출력
    onnx_size = Path(output_path).stat().st_size
    print(f"ONNX model size: {onnx_size / 1024:.1f} KB ({onnx_size / 1024 / 1024:.2f} MB)")


def verify_onnx(model_path: str, onnx_path: str, config_path: str = None):
    """PyTorch vs ONNX 출력 비교 검증."""
    import onnxruntime as ort

    print("\n=== Verification ===")

    # PyTorch 추론
    original_model = load_aasist_model(model_path, config_path)
    inference_model = AASISTInference(original_model)
    inference_model.eval()

    test_input = torch.randn(1, 80000)
    with torch.no_grad():
        pytorch_output = inference_model(test_input).numpy()

    # ONNX 추론
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(None, {"audio": test_input.numpy()})[0]

    # 비교
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

    print(f"PyTorch output: {pytorch_output}")
    print(f"ONNX output:    {onnx_output}")
    print(f"Max diff:  {max_diff:.8f}")
    print(f"Mean diff: {mean_diff:.8f}")

    if max_diff < 1e-4:
        print("PASS: Outputs match within tolerance (1e-4)")
    elif max_diff < 1e-3:
        print("WARN: Small difference (< 1e-3), acceptable for inference")
    else:
        print("FAIL: Outputs differ significantly!")

    return max_diff < 1e-3


def main():
    parser = argparse.ArgumentParser(description="Convert AASIST model to ONNX")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to PyTorch model (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX path (default: same dir as input, .onnx extension)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Model config path (default: AASIST.conf)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX output matches PyTorch",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(args.model_path).with_suffix(".onnx"))

    convert_to_onnx(args.model_path, args.output, args.config)

    if args.verify:
        verify_onnx(args.model_path, args.output, args.config)


if __name__ == "__main__":
    main()
