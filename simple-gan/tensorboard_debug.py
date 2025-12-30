import os

# Check if the directories exist
print("Checking directories...")
print(f"fake exists: {os.path.exists('runs/GAN_SPRITE/fake')}")
print(f"real exists: {os.path.exists('runs/GAN_SPRITE/real')}")

# List files in the directories
if os.path.exists('runs/GAN_SPRITE/fake'):
    print("\nFiles in fake/:")
    print(os.listdir('runs/GAN_SPRITE/fake'))
    
if os.path.exists('runs/GAN_SPRITE/real'):
    print("\nFiles in real/:")
    print(os.listdir('runs/GAN_SPRITE/real'))