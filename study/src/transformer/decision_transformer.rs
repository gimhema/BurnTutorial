use burn::{nn, tensor::{backend::Backend, Tensor}};

#[derive(Debug)]
pub struct DecisionTransformer<B: Backend> {
    embed_timestep: nn::Embedding<B>,
    embed_return: nn::Linear<B>,
    embed_state: nn::Linear<B>,
    embed_action: nn::Linear<B>,
    embed_ln: nn::LayerNorm<B>,
    predict_state: nn::Linear<B>,
    predict_action: nn::Linear<B>,
    predict_return: nn::Linear<B>,
    max_length: Option<usize>,
}

impl<B: Backend> DecisionTransformer<B> {
    pub fn new(device: &B::Device, state_dim: usize, act_dim: usize, hidden_size: usize, max_ep_len: usize, max_length: Option<usize>) -> Self {
        Self {
            embed_timestep: nn::EmbeddingConfig::new(max_ep_len, hidden_size).init(device),
            embed_return: nn::LinearConfig::new(1, hidden_size).init(device),
            embed_state: nn::LinearConfig::new(state_dim, hidden_size).init(device),
            embed_action: nn::LinearConfig::new(act_dim, hidden_size).init(device),
            embed_ln: nn::LayerNormConfig::new(hidden_size).init(device),
            predict_state: nn::LinearConfig::new(hidden_size, state_dim).init(device),
            predict_action: nn::LinearConfig::new(hidden_size, act_dim).init(device),
            predict_return: nn::LinearConfig::new(hidden_size, 1).init(device),
            max_length,
        }
    }

    pub fn forward(
        &self,
        states: Tensor<B, 3>,
        actions: Tensor<B, 3>,
        returns_to_go: Tensor<B, 3>,
        timesteps: Tensor<B, 2, burn::tensor::Int>,
        _attention_mask: Option<Tensor<B, 2, burn::tensor::Int>>
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

    pub fn get_action(
        &self,
        states: Tensor<B, 3>,
        actions: Tensor<B, 3>,
        rewards: Tensor<B, 3>,  // unused variable 경고를 피하기 위해 _rewards로 변경 가능
        returns_to_go: Tensor<B, 3>,
        timesteps: Tensor<B, 2, burn::tensor::Int>
    ) -> Tensor<B, 2> {
        let mut states = states.clone().reshape([1, states.dims()[0], states.dims()[2]]);
        let mut actions = actions.clone().reshape([1, actions.dims()[0], actions.dims()[2]]);
        let mut returns_to_go = returns_to_go.clone().reshape([1, returns_to_go.dims()[0], 1]);
        let mut timesteps = timesteps.clone().reshape([1, timesteps.dims()[0]]);
    
        let attention_mask = if let Some(max_length) = self.max_length {
            let seq_len = states.dims()[1];
            let start_idx = seq_len.saturating_sub(max_length);
    
            states = states.narrow(1, start_idx, max_length);
            actions = actions.narrow(1, start_idx, max_length);
            returns_to_go = returns_to_go.narrow(1, start_idx, max_length);
            timesteps = timesteps.narrow(1, start_idx, max_length);
    
            // mask는 clone()을 사용하여 소유권 문제를 해결
            let mut mask = Tensor::zeros([1, max_length], &states.device());

            Some( mask.slice_assign([max_length.saturating_sub(seq_len)..max_length], Tensor::ones([1, seq_len.saturating_sub(start_idx)], &states.device())) )  
        } else {
            None
        };
    
        let (_, action_preds, _) = self.forward(states, actions, returns_to_go, timesteps, attention_mask);
    
        // action_preds도 clone()을 통해 복사하여 사용
        action_preds.clone().narrow(1, action_preds.dims()[1] - 1, 1).squeeze(1)
    }
    
    
    
}
