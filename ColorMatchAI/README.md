# ColorMatchAI for DaVinci Resolve

ColorMatchAI is an advanced Fusion plugin that utilizes Lua, OpenCL, and Python to perform highly accurate color matching between a reference and a target video. It extracts chromatic data and intelligently generates a 3D LUT which is then applied in real-time via the GPU in DaVinci Resolve.

## Installation Instructions

### Prerequisites
1. **DaVinci Resolve Studio 18/19/20+**
2. **Python 3.9+** installed on your system.
3. Required Python Packages:
   Open a terminal/command prompt and run:
   ```bash
   pip install numpy opencv-python onnxruntime
   ```

### Plugin Installation

You need to place the files from this folder into your DaVinci Resolve Fusion paths.

**For macOS:**
1. Copy `ColorMatchAI.fuse` to:
   `/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Fuses`
   *(or `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Fuses`)*
2. Copy the `Scripts` and `Models` folders to a location of your choice, but ideally within the Fusion Scripts folder:
   `/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/ColorMatchAI/`

**For Windows:**
1. Copy `ColorMatchAI.fuse` to:
   `C:\ProgramData\Blackmagic Design\DaVinci Resolve\Fusion\Fuses`
   *(or `%AppData%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Fuses`)*
2. Copy the `Scripts` and `Models` folders to:
   `C:\ProgramData\Blackmagic Design\DaVinci Resolve\Fusion\Scripts\Utility\ColorMatchAI\`

### How to Use
1. Open DaVinci Resolve and go to the **Fusion Page**.
2. Press `Shift + Spacebar` and search for **ColorMatchAI**.
3. Connect your Target video to the main input (orange/yellow triangle) and your Reference video to the second input (green triangle).
4. Check the **Analyze** box. The plugin will save temporary frames, invoke the Python engine, generate a 3D LUT, and then apply it.
5. Use the Inspector to adjust **Match Strength**, **Preserve Skin**, and toggle the **Scopes**.
6. You can export the generated `.cube` file via the Export button in the Inspector.
