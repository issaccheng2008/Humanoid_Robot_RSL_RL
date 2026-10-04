import os
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf

def generate_ten_sticks_usd(output_path: str):
    """Generate 10 wooden sticks USD for humanoid crossing hurdle course.
    
    Dimensions per user specification:
    - Length (Y, lateral across walking path): 30 cm = 0.30 m
    - Width (X, along walking direction): 3 cm = 0.03 m
    - Height (Z, vertical): 3 cm = 0.03 m
    - Clear gap between adjacent sticks: 20 cm = 0.20 m
    - Pitch (center-to-center): 20 cm + 3 cm = 23 cm = 0.23 m
    - First stick center: X = 0.50 m (front surface at ~0.485 m, ~50 cm from robot start)
    """
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path):
        os.remove(output_path)
        
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    
    # Root Xform
    root = UsdGeom.Xform.Define(stage, "/WoodenBars")
    stage.SetDefaultPrim(root.GetPrim())
    
    # Create Material
    material_path = "/WoodenBars/Looks/WoodMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    # Bright amber wood color for clear visibility against dark terrain
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.88, 0.42, 0.12))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    
    # Stick dimensions: width (X)=0.03m (3cm), length (Y)=0.80m (80cm), height (Z)=0.03m (3cm)
    dx = 0.03
    dy = 0.80
    dz = 0.03
    pitch = 0.23  # 0.20m net gap + 0.03m stick width
    first_x = 0.50
    
    for i in range(10):
        stick_path = f"/WoodenBars/stick_{i}"
        cube = UsdGeom.Cube.Define(stage, stick_path)
        cube.GetSizeAttr().Set(1.0)
        
        # Center position: sitting on the ground (Z=0)
        x_center = first_x + i * pitch
        y_center = 0.0
        z_center = 0.5 * dz
        
        # Scale and translation
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(x_center, y_center, z_center))
        xform.AddScaleOp().Set(Gf.Vec3f(dx, dy, dz))
        
        # Set direct display color as fallback so it's always brightly visible
        cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.88, 0.42, 0.12)])
        
        # Physics collision
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        
        # Bind material
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
        
    stage.GetRootLayer().Save()
    print(f"Successfully generated 10-stick USD at: {output_path}")

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "ten_sticks.usd")
    generate_ten_sticks_usd(out)
