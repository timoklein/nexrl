"""Neural network architectures using Flax NNX for nexrl RL algorithms."""

from nexrl.networks.actor_critic import (
    ActorCritic as ActorCritic,
)
from nexrl.networks.actor_critic import (
    SharedDiscreteActorCritic as SharedDiscreteActorCritic,
)
from nexrl.networks.actor_critic import (
    SharedMLPDiscreteActorCritic as SharedMLPDiscreteActorCritic,
)
from nexrl.networks.base import (
    MLP as MLP,
)
from nexrl.networks.base import (
    FrozenParam as FrozenParam,
)
from nexrl.networks.bbf import (
    BBFNetwork as BBFNetwork,
)
from nexrl.networks.dreamer import (
    RSSM as RSSM,
)
from nexrl.networks.dreamer import (
    ContinuePredictor as ContinuePredictor,
)
from nexrl.networks.dreamer import (
    DreamerActor as DreamerActor,
)
from nexrl.networks.dreamer import (
    DreamerCritic as DreamerCritic,
)
from nexrl.networks.dreamer import (
    DreamerDecoder as DreamerDecoder,
)
from nexrl.networks.dreamer import (
    DreamerEncoder as DreamerEncoder,
)
from nexrl.networks.dreamer import (
    DreamerWorldModel as DreamerWorldModel,
)
from nexrl.networks.dreamer import (
    GRUCell as GRUCell,
)
from nexrl.networks.dreamer import (
    RewardPredictor as RewardPredictor,
)
from nexrl.networks.drqv2 import (
    DrQV2Actor as DrQV2Actor,
)
from nexrl.networks.drqv2 import (
    DrQV2Critic as DrQV2Critic,
)
from nexrl.networks.drqv2 import (
    DrQV2Encoder as DrQV2Encoder,
)
from nexrl.networks.encoder import (
    ImpalaCNNEncoder as ImpalaCNNEncoder,
)
from nexrl.networks.encoder import (
    NatureCNNEncoder as NatureCNNEncoder,
)
from nexrl.networks.exploration import (
    ICMModule as ICMModule,
)
from nexrl.networks.exploration import (
    RNDModule as RNDModule,
)
from nexrl.networks.fast_sac import (
    FastSACCritic as FastSACCritic,
)
from nexrl.networks.fast_sac import (
    FastSACPolicy as FastSACPolicy,
)
from nexrl.networks.fast_td3 import (
    FastTD3Critic as FastTD3Critic,
)
from nexrl.networks.fast_td3 import (
    FastTD3Policy as FastTD3Policy,
)
from nexrl.networks.flash_sac import (
    FlashSACCritic as FlashSACCritic,
)
from nexrl.networks.flash_sac import (
    FlashSACPolicy as FlashSACPolicy,
)
from nexrl.networks.flash_sac import (
    project_params as project_params,
)
from nexrl.networks.mrq import (
    MrQEncoder as MrQEncoder,
)
from nexrl.networks.mrq import (
    MrQPolicy as MrQPolicy,
)
from nexrl.networks.mrq import (
    MrQValue as MrQValue,
)
from nexrl.networks.mrsq import (
    MrsQEncoder as MrsQEncoder,
)
from nexrl.networks.mrsq import (
    MrsQValue as MrsQValue,
)
from nexrl.networks.policies import (
    DeterministicPolicy as DeterministicPolicy,
)
from nexrl.networks.policies import (
    DiscretePolicy as DiscretePolicy,
)
from nexrl.networks.policies import (
    GaussianPolicy as GaussianPolicy,
)
from nexrl.networks.policies import (
    SquashedGaussianPolicy as SquashedGaussianPolicy,
)
from nexrl.networks.rnn import (
    ActorCriticRNN as ActorCriticRNN,
)
from nexrl.networks.rnn import (
    StandardGRUCell as StandardGRUCell,
)
from nexrl.networks.sgf import (
    SGFEncoder as SGFEncoder,
)
from nexrl.networks.sgf import (
    SGFPredictor as SGFPredictor,
)
from nexrl.networks.sgf import (
    SGFProjector as SGFProjector,
)
from nexrl.networks.sgf import (
    SGFTransitionPredictor as SGFTransitionPredictor,
)
from nexrl.networks.sgf import (
    SGFWorldModel as SGFWorldModel,
)
from nexrl.networks.simba import (
    ResidualBlock as ResidualBlock,
)
from nexrl.networks.simba import (
    SimBaEncoder as SimBaEncoder,
)
from nexrl.networks.simba import (
    SimBaQNetwork as SimBaQNetwork,
)
from nexrl.networks.simba import (
    SimBaSquashedGaussianPolicy as SimBaSquashedGaussianPolicy,
)
from nexrl.networks.value import (
    CategoricalQNetwork as CategoricalQNetwork,
)
from nexrl.networks.value import (
    DiscreteQNetwork as DiscreteQNetwork,
)
from nexrl.networks.value import (
    DuelingQNetwork as DuelingQNetwork,
)
from nexrl.networks.value import (
    ImplicitQuantileNetwork as ImplicitQuantileNetwork,
)
from nexrl.networks.value import (
    NormalizedDiscreteQNetwork as NormalizedDiscreteQNetwork,
)
from nexrl.networks.value import (
    QNetwork as QNetwork,
)
from nexrl.networks.value import (
    VNetwork as VNetwork,
)
from nexrl.networks.world_model import (
    DynamicsNetwork as DynamicsNetwork,
)
from nexrl.networks.world_model import (
    MuZeroModel as MuZeroModel,
)
from nexrl.networks.world_model import (
    PredictionNetwork as PredictionNetwork,
)
from nexrl.networks.world_model import (
    RepresentationNetwork as RepresentationNetwork,
)
