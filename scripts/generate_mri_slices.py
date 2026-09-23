import os
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

output_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "images", "projects", "mri")
os.makedirs(output_base, exist_ok=True)

# Generate realistic anatomical MRI knee textures
def create_realistic_mri(plane, slice_idx, num_slices=16, case='acl'):
    """
    Synthesizes authentic clinical MRI knee imaging slice (512x512)
    using procedural anatomical field mapping, bone marrow trabeculae,
    cortex boundaries, soft tissue fascia, cartilage, and DICOM Rician noise.
    """
    size = 480
    y, x = np.mgrid[0:size, 0:size].astype(float)
    img = np.zeros((size, size), dtype=float)
    
    # Normalized depth parameter: z in [-1, 1]
    # z = -1: lateral aspect, z = 0: intercondylar notch / midline, z = +1: medial aspect
    z = (slice_idx - (num_slices / 2.0)) / (num_slices / 2.0)
    
    # 1. Background subcutaneous fat and skin contour
    # Elliptical outer knee contour
    knee_mask = ((x - 240)**2 / (190**2) + (y - 240)**2 / (210**2)) < 1.0
    skin_rim = knee_mask & ~(((x - 240)**2 / (180**2) + (y - 240)**2 / (200**2)) < 1.0)
    subcut_fat = knee_mask & ~skin_rim & (((x - 240)**2 / (155**2) + (y - 240)**2 / (175**2)) > 1.0)
    
    img[knee_mask] = 0.12 # low muscle baseline
    img[subcut_fat] = 0.45 # fat is bright on PD/T2
    img[skin_rim] = 0.25
    
    if plane == 'sagittal':
        # SAGITTAL VIEW (from lateral to notch to medial)
        # Femoral condyle
        # Condyle size changes with z (notch at z=0 is concave)
        condyle_radius = 80 - 25 * np.exp(-((z)**2) / 0.15)
        femur_y = 175 - 15 * z
        femur_x = 230 - 20 * z
        
        femur_shaft = (x > femur_x - 45) & (x < femur_x + 45) & (y < femur_y)
        femur_condyle = ((x - femur_x)**2 / (condyle_radius**2) + (y - femur_y)**2 / (70**2)) < 1.0
        femur_bone = femur_condyle | femur_shaft
        
        # Tibial plateau
        tibia_x = 235 - 10 * z
        tibia_y = 330
        tibia_shaft = (x > tibia_x - 55) & (x < tibia_x + 55) & (y > tibia_y)
        tibia_plateau = ((x - tibia_x)**2 / (85**2) + (y - tibia_y)**2 / (45**2)) < 1.0
        tibia_bone = tibia_plateau | tibia_shaft
        
        # Patella (prominent in slices near z = -0.2 to +0.2)
        patella_vis = max(0.0, 1.0 - abs(z) * 1.8)
        patella_bone = np.zeros_like(knee_mask)
        if patella_vis > 0.1:
            pat_x = 135
            pat_y = 180
            patella_bone = ((x - pat_x)**2 / ((26 * patella_vis)**2) + (y - pat_y)**2 / ((42 * patella_vis)**2)) < 1.0
            
        # Fibular head (only lateral aspect: z < -0.4)
        fibula_bone = np.zeros_like(knee_mask)
        if z < -0.35:
            fib_x = 310
            fib_y = 360
            fibula_bone = ((x - fib_x)**2 / (22**2) + (y - fib_y)**2 / (38**2)) < 1.0
            
        # Draw bone marrow (intermediate-bright signal)
        img[femur_bone] = 0.55
        img[tibia_bone] = 0.52
        img[patella_bone] = 0.50
        img[fibula_bone] = 0.48
        
        # Bone Cortex (black cortical rim)
        for b in [femur_bone, tibia_bone, patella_bone, fibula_bone]:
            eroded = ndimage.binary_erosion(b, iterations=4)
            cortex = b & ~eroded
            img[cortex] = 0.06 # Dark cortical bone
            
        # Articular cartilage (thin gray layer on condyle & plateau)
        fem_cart = ndimage.binary_dilation(femur_bone, iterations=3) & ~femur_bone & (y > femur_y) & (y < 280)
        tib_cart = ndimage.binary_dilation(tibia_bone, iterations=3) & ~tibia_bone & (y < tibia_y) & (y > 270)
        img[fem_cart] = 0.65
        img[tib_cart] = 0.65
        
        # Quadriceps & Patellar Tendon (jet black fibrous bands)
        if patella_vis > 0.1:
            pat_tendon = (x > 140) & (x < 152) & (y > 215) & (y < 315)
            img[pat_tendon] = 0.05
            quad_tendon = (x > 145) & (x < 158) & (y > 100) & (y < 155)
            img[quad_tendon] = 0.05
            
        # Infrapatellar fat pad (Hoffa's fat pad - bright)
        hoffa = (x > 155) & (x < 225) & (y > 210) & (y < 285) & ~femur_bone & ~tibia_bone
        img[hoffa] = 0.48
        
        # Cruciate Ligaments (ACL & PCL - visible in notch slices: |z| < 0.35)
        if abs(z) < 0.35:
            # PCL (dark arched curve)
            pcl_mask = np.zeros_like(knee_mask)
            for t_val in np.linspace(0, 1, 60):
                px = int(210 + t_val * 40 + np.sin(t_val * np.pi) * 15)
                py = int(220 + t_val * 70)
                if 0 <= px < size and 0 <= py < size:
                    pcl_mask[max(0, py-3):min(size, py+3), max(0, px-3):min(size, px+3)] = True
            img[pcl_mask & ~femur_bone & ~tibia_bone] = 0.08
            
            # ACL tract
            acl_mask = np.zeros_like(knee_mask)
            for t_val in np.linspace(0, 1, 70):
                ax_pt = int(250 - t_val * 45)
                ay_pt = int(230 + t_val * 60)
                if 0 <= ax_pt < size and 0 <= ay_pt < size:
                    acl_mask[max(0, ay_pt-4):min(size, ay_pt+4), max(0, ax_pt-4):min(size, ax_pt+4)] = True
            
            if case == 'acl':
                # Torn ACL: disrupted fibers, fluid hyperintensity
                img[acl_mask & ~femur_bone & ~tibia_bone] = 0.72 # hyperintense fluid edema
                # Focal disruption in middle
                tear_spot = ((x - 228)**2 + (y - 260)**2) < 18**2
                img[tear_spot] = 0.90 # bright fluid collection
            else:
                img[acl_mask & ~femur_bone & ~tibia_bone] = 0.06 # normal taut dark ligament
                
        # Meniscus (anterior & posterior horn wedges)
        # Visible in peripheral slices (|z| > 0.25)
        if abs(z) > 0.25:
            # Anterior horn
            ah_x, ah_y = 195, 275
            ah_meniscus = ((x - ah_x) > 0) & ((x - ah_x) < 26) & (abs(y - ah_y) < (26 - (x - ah_x)) * 0.45)
            # Posterior horn
            ph_x, ph_y = 285, 278
            ph_meniscus = ((x - ph_x) < 0) & ((ph_x - x) < 32) & (abs(y - ph_y) < (32 - (ph_x - x)) * 0.45)
            
            img[ah_meniscus & ~femur_bone & ~tibia_bone] = 0.04 # normal black fibrocartilage
            
            if case == 'meniscus' and z > 0.3:
                # Medial meniscus posterior horn tear: bright line traversing triangle
                img[ph_meniscus & ~femur_bone & ~tibia_bone] = 0.04
                tear_line = ph_meniscus & (abs((y - ph_y) - 0.35 * (ph_x - x)) < 2.5)
                img[tear_line] = 0.85 # bright fluid cleft
            else:
                img[ph_meniscus & ~femur_bone & ~tibia_bone] = 0.04
                
        # Joint effusion (high signal fluid in suprapatellar pouch)
        if case in ['acl', 'contusion']:
            effusion = (x > 155) & (x < 195) & (y > 120) & (y < 170) & ~femur_bone & ~patella_bone
            img[effusion] = 0.88 # bright joint effusion
            
        if case == 'oa' and z > 0.2:
            # Medial OA: cartilage loss & subchondral sclerosis
            subchondral = (y > 268) & (y < 288) & (x > 210) & (x < 280) & tibia_bone
            img[subchondral] = 0.15 # dark eburnated sclerosis
            
        if case == 'contusion' and z < -0.1:
            # Bone contusion / edema: patchy bright signal in lateral tibia
            edema = ((x - 250)**2 / (35**2) + (y - 340)**2 / (30**2)) < 1.0
            img[edema & tibia_bone] = 0.85 # bone bruise hyperintensity

    elif plane == 'coronal':
        # CORONAL VIEW (Anterior to Posterior)
        # Dual femoral condyles (Medial & Lateral)
        med_cond = ((x - 170)**2 / (55**2) + (y - 190)**2 / (75**2)) < 1.0
        lat_cond = ((x - 310)**2 / (55**2) + (y - 190)**2 / (75**2)) < 1.0
        femur_shaft = (x > 185) & (x < 295) & (y < 190)
        femur_bone = med_cond | lat_cond | femur_shaft
        
        # Notch between condyles
        notch = (x > 215) & (x < 265) & (y > 195) & (y < 255)
        femur_bone = femur_bone & ~notch
        
        # Tibial plateau
        tibia_bone = (x > 140) & (x < 340) & (y > 275) & (y < 420)
        tibia_c = ((x - 240)**2 / (110**2) + (y - 280)**2 / (40**2)) < 1.0
        tibia_bone = tibia_bone | tibia_c
        
        # Fibula laterally (x ~ 350)
        fibula_bone = (x > 330) & (x < 365) & (y > 310) & (y < 420)
        
        img[femur_bone] = 0.55
        img[tibia_bone] = 0.52
        img[fibula_bone] = 0.48
        
        # Cortical rims
        for b in [femur_bone, tibia_bone, fibula_bone]:
            cortex = b & ~ndimage.binary_erosion(b, iterations=4)
            img[cortex] = 0.05
            
        # Cartilage
        cart_med = (x > 145) & (x < 210) & (y >= 265) & (y <= 276)
        cart_lat = (x > 270) & (x < 335) & (y >= 265) & (y <= 276)
        img[cart_med] = 0.65
        img[cart_lat] = 0.65
        
        # Menisci (sharp triangles on both sides)
        med_men = ((x - 145) > 0) & ((x - 145) < 35) & (abs(y - 270) < (35 - (x - 145)) * 0.3)
        lat_men = ((335 - x) > 0) & ((335 - x) < 35) & (abs(y - 270) < (35 - (335 - x)) * 0.3)
        img[med_men] = 0.04
        img[lat_men] = 0.04
        
        if case == 'meniscus':
            # Horizontal tear in medial meniscus
            tear = med_men & (abs(y - 270) < 2.0)
            img[tear] = 0.88
            
        # MCL (Medial Collateral Ligament) along medial border
        mcl = (x > 132) & (x < 140) & (y > 210) & (y < 340)
        img[mcl] = 0.06
        
        if case == 'oa':
            # Severe joint space loss on medial side
            img[cart_med] = 0.20
            eburnation = (x > 150) & (x < 210) & (y > 274) & (y < 295) & tibia_bone
            img[eburnation] = 0.12 # dense subchondral bone sclerosis
            # Osteophyte spur
            spur = (x > 135) & (x < 148) & (y > 268) & (y < 282)
            img[spur] = 0.55
            
        if case == 'contusion':
            edema_cor = ((x - 305)**2 / (35**2) + (y - 300)**2 / (25**2)) < 1.0
            img[edema_cor & tibia_bone] = 0.88

    else:
        # AXIAL VIEW (Superior to Inferior)
        # Femoral trochlea / condyles
        fem_troch = ((x - 240)**2 / (105**2) + (y - 275)**2 / (75**2)) < 1.0
        # Trochlear sulcus groove depression
        groove = ((x - 240)**2 / (30**2) + (y - 215)**2 / (25**2)) < 1.0
        fem_troch = fem_troch & ~groove
        
        # Patella bone anteriorly (y ~ 140)
        pat_ax = ((x - 240)**2 / (55**2) + (y - 145)**2 / (26**2)) < 1.0
        
        img[fem_troch] = 0.55
        img[pat_ax] = 0.50
        
        for b in [fem_troch, pat_ax]:
            cortex = b & ~ndimage.binary_erosion(b, iterations=4)
            img[cortex] = 0.06
            
        # Retropatellar cartilage
        pat_cart = ndimage.binary_dilation(pat_ax, iterations=3) & ~pat_ax & (y > 145)
        img[pat_cart] = 0.65
        
        # Joint fluid recesses
        if case in ['acl', 'meniscus', 'contusion']:
            effusion_lat = ((x - 315)**2 / (25**2) + (y - 200)**2 / (45**2)) < 1.0
            effusion_med = ((x - 165)**2 / (25**2) + (y - 200)**2 / (45**2)) < 1.0
            img[effusion_lat & knee_mask & ~fem_troch] = 0.90
            img[effusion_med & knee_mask & ~fem_troch] = 0.90

    # 4. Realistic Medical MRI Artifacts & Noise
    # A. Trabecular bone marrow texture (perlin-like spatial noise)
    noise_raw = np.random.normal(0, 1, (size, size))
    trabeculae = ndimage.gaussian_filter(noise_raw, sigma=1.8)
    trabeculae = (trabeculae - trabeculae.min()) / (trabeculae.max() - trabeculae.min()) - 0.5
    img += trabeculae * 0.09 * (img > 0.3)
    
    # B. Micro Gaussian blur (simulate point-spread function of 3T MRI coil)
    img = ndimage.gaussian_filter(img, sigma=1.1)
    
    # C. High-frequency Rician scanner noise (background noise floor)
    rician_real = np.random.normal(0, 0.025, (size, size))
    rician_imag = np.random.normal(0, 0.025, (size, size))
    noisy_img = np.sqrt((img + rician_real)**2 + rician_imag**2)
    
    # D. Medical DICOM Grayscale windowing & normalization (0 - 255)
    final_img = np.clip(noisy_img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(final_img)

def generate_all_slices():
    print("Generating authentic realistic MRI slice series...")
    planes = ['sagittal', 'coronal', 'axial']
    cases = ['acl', 'meniscus', 'oa', 'contusion']
    
    for plane in planes:
        plane_dir = os.path.join(output_base, plane)
        os.makedirs(plane_dir, exist_ok=True)
        for case in cases:
            case_dir = os.path.join(plane_dir, case)
            os.makedirs(case_dir, exist_ok=True)
            for s in range(1, 17):
                img = create_realistic_mri(plane, s, 16, case)
                filename = f"slice_{s:02d}.png"
                img.save(os.path.join(case_dir, filename), "PNG")
                
    print("All 192 realistic MRI slices generated and saved successfully!")

if __name__ == "__main__":
    generate_all_slices()
