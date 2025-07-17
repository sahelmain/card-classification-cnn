import os

def verify_dataset():
    data_path = 'data'
    print(f"Looking for data in: {os.path.abspath(data_path)}")
    print(f"Current working directory: {os.getcwd()}")
    
    if not os.path.exists(data_path):
        print(f"❌ Data path {data_path} doesn't exist")
        return False
    
    required_dirs = ['train', 'valid', 'test']
    
    for split in required_dirs:
        split_path = os.path.join(data_path, split)
        print(f"Checking: {split_path}")
        if not os.path.exists(split_path):
            print(f"❌ Missing {split} directory at {split_path}")
            return False
        
        classes = os.listdir(split_path)
        classes = [c for c in classes if os.path.isdir(os.path.join(split_path, c))]
        
        print(f"✅ {split}: {len(classes)} classes found")
        
        # Count images
        total_images = 0
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(images)
        
        print(f"   📸 Total images in {split}: {total_images}")
    
    print("✅ Dataset structure looks good!")
    return True

if __name__ == "__main__":
    verify_dataset() 