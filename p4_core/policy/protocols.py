from abc import ABC, abstractmethod

from .models import Completion, Prompt, SystemMessage


class BaseChatPolicy[Observation, Action](ABC):
    system_message: SystemMessage

    def act(
        self, observation: Observation, previous_observation: Observation
    ) -> Action:
        prompt = self.prompt_from_observation(observation, previous_observation)
        inputs = [prompt]
        response = self.completions_from_prompts(inputs)[0]
        return self.action_from_completion(response, prompt)

    @abstractmethod
    def prompt_from_observation(
        self, observation: Observation, previous_observation: Observation
    ) -> Prompt: ...

    @abstractmethod
    def completions_from_prompts(
        self,
        prompts: list[Prompt],
    ) -> list[Completion]: ...

    @abstractmethod
    def action_from_completion(
        self, completion: Completion, prompt: Prompt
    ) -> Action: ...
