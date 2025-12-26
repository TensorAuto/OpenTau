import pytest

from lerobot.common.planner import Memory


def add_conversation_for_test(mem: Memory, no_of_conversation: int) -> Memory:
    """
    Adds conversation using add_conversation method in Memory class

    Args:
        mem: Memory instance
        no_of_conversations: number of conversations to be added

    Returns:
        mem: Updated Memory instance
    """

    for _ in range(no_of_conversation):
        mem.add_conversation(role="user", message=[{"text": "This is test message"}])

    return mem


@pytest.mark.parametrize("mem", [None, 10], indirect=True)
def test_default_values(mem):
    """
    Checks if the variables are properly being initialized by setting value or using default value
    """

    buffer = mem["object"].len
    conversation = mem["object"].conversation

    assert buffer == mem["len"]
    assert len(conversation) == 1


@pytest.mark.parametrize("mem", [None, 10], indirect=True)
def test_add_conversation(mem):
    """
    Checks if the add_conversation method is working properly by adding few conversation less than buffer length
    """

    length = mem["len"]

    mem["object"] = add_conversation_for_test(mem["object"], length // 2)

    assert len(mem["object"].conversation) == (length // 2) + 1


@pytest.mark.parametrize("mem", [None, 10], indirect=True)
def test_buffer_limit(mem):
    """
    Checks if the add_conversation method is working properly by adding few conversation more than buffer length
    """

    length = mem["len"]

    mem["object"] = add_conversation_for_test(mem["object"], length + 10)

    assert len(mem["object"].conversation) == length


@pytest.mark.parametrize("mem", [None, 10], indirect=True)
def test_get_conversation(mem):
    """
    Checks if the get_convcersation returns the same conversation variable stored in mem instance
    """

    mem["object"] = add_conversation_for_test(mem["object"], 5)

    conversation = mem["object"].get_conversation()

    assert conversation == list(mem["object"].conversation)
