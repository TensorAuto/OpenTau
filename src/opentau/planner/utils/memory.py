# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import deque

from opentau.planner.utils.utils import load_prompt_library


class Memory:
    """
    Generates a memory like class to store, retrieve all the conversations for a particular task between user and LLM assistant
    """

    def __init__(self, conversation=None, len=1000):
        """
        Initializes conservation variable if any earlier conservation needs to be append before initilaizing the memory object.
        len: fixed buffer size of conversation
        """

        self.len = len
        self.prompts_dict = load_prompt_library("src/opentau/planner/prompts.yaml")

        if conversation:
            self.conversation = conversation
        else:
            self.conversation = deque()
            context = [
                {
                    "type": "text",
                    "text": f"{self.prompts_dict['prompts']['robot_system_manipulation']['template']}",
                }
            ]
            self.add_conversation("system", context)

    def add_conversation(self, role: str, message: list[dict[str, str]]) -> None:
        """
        Adds new conversations to history of conversations.

        Args:
            role (str): The message given by (system, user, assistant)
            message (list[dict[str, str]]): message containing text and/or images.
        """

        if len(self.conversation) >= self.len:
            self.conversation.popleft()

        self.conversation.append({"role": role, "content": message})

    def get_conversation(self) -> list[dict[str, str]]:
        """
        Returns the stored conversation or history of conversations
        """
        return list(self.conversation)
