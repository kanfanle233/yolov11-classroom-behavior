import os, json

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = os.path.join(base_dir, "output", "pose_keypoints_v2.jsonl")


    with open(in_path, "r", encoding="utf-8") as f:
        for i in range(3):
            line = f.readline().strip()
            obj = json.loads(line)
            print("Top keys:", obj.keys())
            persons = obj.get("persons", [])
            print("persons len:", len(persons))
            if persons:
                print("person keys:", persons[0].keys())
                print("sample person:", persons[0])
            print("-" * 60)

if __name__ == "__main__":
    main()
