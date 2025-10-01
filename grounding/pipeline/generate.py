from ..datasets.data import GroundingData
from ..format.detect_single_formater import DetectSingleFormater

if __name__ == "__main__":
    formater = DetectSingleFormater()
    with open(
        "/mnt/aigc/users/pufanyi/workspace/grounding-data/data/result_dataset.jsonl"
    ) as f:
        num = 0
        for line in f:
            data = GroundingData.model_validate_json(line)
            if formater.check_eligible(data):
                print(formater.format(data))
                num += 1
                if num > 10:
                    break
