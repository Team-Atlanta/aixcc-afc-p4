from abc import ABC, abstractmethod
from typing import Callable

from joblib import Parallel, delayed

from ..policy.models import Completion, Prompt


class BaseEnvironment[Observation, Action, Context](ABC):
    def step(
        self,
        action: Action,
        observation: Observation,
        context: Context,
    ):
        return self._step(action, observation, context)

    @abstractmethod
    def _step(
        self,
        action: Action,
        observation: Observation,
        context: Context,
    ) -> tuple[Observation, bool, bool]: ...  # [Observation,  Terminated, Truncated]

    @abstractmethod
    def reset(self, context: Context) -> Observation: ...

    def as_trainable_with_chat(
        self,
        immediate_reward_functions: list[
            Callable[[Observation, Action, Context, Prompt, Completion], float]
        ],
        final_reward_functions: list[
            Callable[[Observation, Action, Context, Prompt, Completion], float]
        ],
        immediate_reward_weights: list[float] | None = None,
        final_reward_weights: list[float] | None = None,
    ):
        return BaseTrainableEnvironmentWithChat(
            self,
            immediate_reward_functions,
            final_reward_functions,
            immediate_reward_weights,
            final_reward_weights,
        )


class BaseTrainableEnvironmentWithChat[Observation, Action, Context]:
    def __init__(
        self,
        environment: BaseEnvironment[Observation, Action, Context],
        immediate_reward_functions: list[
            Callable[[Observation, Action, Context, Prompt, Completion], float]
        ],
        final_reward_functions: list[
            Callable[[Observation, Action, Context, Prompt, Completion], float]
        ],
        immediate_reward_weights: list[float] | None = None,
        final_reward_weights: list[float] | None = None,
    ):
        if immediate_reward_weights is None:
            immediate_reward_weights = [1.0] * len(immediate_reward_functions)
        if final_reward_weights is None:
            final_reward_weights = [1.0] * len(final_reward_functions)

        if len(immediate_reward_weights) != len(immediate_reward_functions):
            raise ValueError(
                "immediate_reward_weights must be the same length as immediate_reward_functions"
            )
        if len(final_reward_weights) != len(final_reward_functions):
            raise ValueError(
                "final_reward_weights must be the same length as final_reward_functions"
            )

        self._environment = environment
        self._immediate_reward_functions = immediate_reward_functions
        self._final_reward_functions = final_reward_functions
        self._immediate_reward_weights = immediate_reward_weights
        self._final_reward_weights = final_reward_weights

    @property
    def reward_functions(self):
        return self._immediate_reward_functions + self._final_reward_functions

    @property
    def reward_weights(self):
        return self._immediate_reward_weights + self._final_reward_weights

    def batch(
        self,
        actions: list[Action],
        observations: list[Observation],
        contexts: list[Context],
        prompts: list[Prompt],
        completions: list[Completion],
    ):
        return Parallel(n_jobs=-1, backend="threading")(
            delayed(self.step)(
                action,
                observation,
                context,
                prompt,
                completion,
            )
            for action, observation, context, prompt, completion in zip(
                actions, observations, contexts, prompts, completions
            )
        )

    def step(
        self,
        action: Action,
        observation: Observation,
        context: Context,
        prompt: Prompt,
        completion: Completion,
    ):
        observation, terminated, truncated = self._environment.step(
            action, observation, context
        )

        if terminated or truncated:
            rewards: list[float | None] = [
                reward_function(observation, action, context, prompt, completion)
                for reward_function in self.reward_functions
            ]
        else:
            rewards: list[float | None] = [
                reward_function(observation, action, context, prompt, completion)
                for reward_function in self._immediate_reward_functions
            ] + [None] * len(self._final_reward_functions)

        return observation, rewards, terminated, truncated

    @abstractmethod
    def reset(self, context: Context) -> Observation:
        return self._environment.reset(context)
