from pathlib import Path

from src.dialogue_system.utils.global_variables import RESOURCES_PATH


def create_session_folder(reward, chat_id, speaker):
    # Create folder to store session
    session_folder = Path(RESOURCES_PATH +
                          f"{reward.replace(' ', '-')}_"
                          f"{chat_id}_"
                          f"{speaker.replace(' ', '-')}/")
    session_folder.mkdir(parents=True, exist_ok=True)

    return session_folder
