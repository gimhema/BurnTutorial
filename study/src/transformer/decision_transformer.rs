use burn::{nn, tensor::{backend::Backend, Tensor}};
use burn::module::Module;

#[derive(Module, Debug)]
pub struct DecisionTransformer<B: Backend> {
    embed_timestep: nn::Embedding<B>,
    embed_return: nn::Linear<B>,
    embed_state: nn::Linear<B>,
    embed_action: nn::Linear<B>,
    embed_ln: nn::LayerNorm<B>,
    predict_state: nn::Linear<B>,
    predict_action: nn::Linear<B>,
    predict_return: nn::Linear<B>,
}

impl<B: Backend> DecisionTransformer<B> {
    pub fn new(device: &B::Device, state_dim: usize, act_dim: usize, hidden_size: usize, max_ep_len: usize) -> Self {
        Self {
            embed_timestep: nn::EmbeddingConfig::new(max_ep_len, hidden_size).init(device),
            embed_return: nn::LinearConfig::new(1, hidden_size).init(device),
            embed_state: nn::LinearConfig::new(state_dim, hidden_size).init(device),
            embed_action: nn::LinearConfig::new(act_dim, hidden_size).init(device),
            embed_ln: nn::LayerNormConfig::new(hidden_size).init(device),
            predict_state: nn::LinearConfig::new(hidden_size, state_dim).init(device),
            predict_action: nn::LinearConfig::new(hidden_size, act_dim).init(device),
            predict_return: nn::LinearConfig::new(hidden_size, 1).init(device),
        }
    }

    pub fn forward(
        &self,
        states: Tensor<B, 3>,
        actions: Tensor<B, 3>,
        returns_to_go: Tensor<B, 3>,
        timesteps: Tensor<B, 2, burn::tensor::Int>
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let state_embeddings = self.embed_state.forward(states);
        let action_embeddings = self.embed_action.forward(actions);
        let returns_embeddings = self.embed_return.forward(returns_to_go);
        let time_embeddings = self.embed_timestep.forward(timesteps);

        let stacked_inputs = state_embeddings + time_embeddings.clone() + action_embeddings + returns_embeddings;
        let stacked_inputs = self.embed_ln.forward(stacked_inputs);

        let return_preds = self.predict_return.forward(stacked_inputs.clone());
        let state_preds = self.predict_state.forward(stacked_inputs.clone());
        let action_preds = self.predict_action.forward(stacked_inputs);

        (state_preds, action_preds, return_preds)
    }
}
