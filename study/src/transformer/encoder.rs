use burn::backend::ndarray::NdArray;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Shape};
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn::module::Module;
use burn::config::Config;


pub struct Encoder
{
    
}

impl Encoder
{
    pub fn create() {
        type Backend = NdArray<f32>;

    // Transformer 인코더 구성 설정
    let config = TransformerEncoderConfig::new(4, 8, 64) // (레이어 개수, 헤드 수, 임베딩 차원)
        .with_dropout(0.1);

    // Transformer 인코더 생성
    let encoder = config.init();

    // 임의의 입력 텐서 생성 (배치 크기 2, 토큰 개수 5, 차원 64)
    let input_tensor = Tensor::<Backend, 3>::random(Shape::new([2, 5, 64]), 0.0, 1.0);

    // Transformer 인코더 입력 객체 생성
    let encoder_input = TransformerEncoderInput::new(input_tensor);

    // Transformer 인코더 실행
    let output = encoder.forward(encoder_input);

    // 출력 확인
    println!("Transformer Encoder Output: {:?}", output);
    }
}
