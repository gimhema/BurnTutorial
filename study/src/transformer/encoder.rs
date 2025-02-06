use burn::backend::ndarray::NdArray;
use burn::tensor::{Tensor, Shape, Distribution};
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn::module::Module;
use burn::config::Config;
use burn::tensor::backend::Backend;

// 백엔드 타입 선언
type MyBackend = NdArray;

pub struct Encoder<B: Backend> {
    encoder: TransformerEncoder<B>,
    device: B::Device,
}

impl<B: Backend> Encoder<B> {
    pub fn new() -> Self {
        let device = B::Device::default(); // 디바이스 초기화

        // Transformer 인코더 구성 설정
        // let config = TransformerEncoderConfig::new(4, 8, 64, 4) // (레이어 개수, 헤드 수, 임베딩 차원)
        //     .with_dropout(0.1);
        let config = TransformerEncoderConfig::new(64, 8, 4, 256).with_dropout(0.1);

        // Transformer 인코더 생성
        let encoder = config.init::<B>(&device);

        Self { encoder, device }
    }

    pub fn encode(&self) {
        // 임의의 입력 텐서 생성 (배치 크기 2, 토큰 개수 5, 차원 64)
        let input_tensor = Tensor::<B, 3>::random(Shape::new([2, 5, 64]), Distribution::Uniform(0.0, 1.0), &self.device);

        println!("Input Tensor Shape: {:?}", input_tensor.shape());

        // Transformer 인코더 입력 객체 생성
        let encoder_input = TransformerEncoderInput::new(input_tensor);

        // Transformer 인코더 실행
        let output = self.encoder.forward(encoder_input);

        // 출력 확인
        println!("Transformer Encoder Output: {:?}", output);
    }

}

pub fn encode_test() {
    let encoder = Encoder::<MyBackend>::new();
    encoder.encode();
}

// fn main() {
//     let encoder = Encoder::<MyBackend>::new();
//     encoder.encode();
// }
