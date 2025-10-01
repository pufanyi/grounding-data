from ..datasets.data import GroundingData
from ..format.detect_multi_object_formatter import DetectMultiObjectFormatter

if __name__ == "__main__":
    # formatter = DetectSingleFormatter()
    # formatter = DetectMultiFormatter()
    formatter = DetectMultiObjectFormatter()
    with open(
        "/mnt/aigc/users/pufanyi/workspace/grounding-data/data/result_dataset.jsonl"
    ) as f:
        num = 0
        for line in f:
            data = GroundingData.model_validate_json(line)
            if formatter.check_eligible(data):
                print(formatter.format(data))
                num += 1
                if num > 10:
                    break
