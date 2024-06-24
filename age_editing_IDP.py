import os
import argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler
from pytorch_lightning import seed_everything

from FADING_util import util
from p2p import *
from null_inversion import *


UNIQUE_TOKEN = "sks"


def run():
    args = parse_args()
    image_path = args.image_path
    age_init = args.age_init
    gender = args.gender
    specialized_path = args.specialized_path
    target_ages = args.target_ages
    save_aged_dir = args.save_aged_dir
    if not os.path.exists(save_aged_dir):
        os.makedirs(save_aged_dir)
    seed_everything(args.seed)

    input_img_name = image_path.split('/')[-1].split('.')[-2]

    ldm_stable, g_cuda, tokenizer = load_diffusers(specialized_path)

    age_grouped = classify_age(age_init)
    gt_gender = int(gender == 'female')
    person_placeholder = util.get_person_placeholder(age_init, gt_gender)
    inversion_prompt = f"photo of {UNIQUE_TOKEN} {person_placeholder} as {age_grouped}"
    # inversion_prompt = f"photo of {UNIQUE_TOKEN} person as {age_grouped}"
    x_t, uncond_embeddings = invert(ldm_stable, image_path, inversion_prompt)
    
    #! age editing
    for age_new in target_ages:
        age_grouped_new = classify_age(age_new)
        print(f'Age editing with target age {age_new} ({age_grouped_new})...')
        
        new_person_placeholder = util.get_person_placeholder(age_new, gt_gender)
        prompt_before_after = ((str(age_grouped), person_placeholder), (str(age_grouped_new), new_person_placeholder))
        images = prompt_to_prompt(inversion_prompt, prompt_before_after,
                                  ldm_stable, g_cuda, tokenizer,
                                  x_t, uncond_embeddings,)

        output_img_name = f'{input_img_name}_{age_new}_{age_grouped_new}.png'
        save_output(images, save_aged_dir, output_img_name)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', required=True, help='Path to input image')
    parser.add_argument('--age_init', required=True, type=int, help='Specify the initial age')
    parser.add_argument('--gender', choices=["female", "male"], help="Specify the gender ('female' or 'male')")
    parser.add_argument('--specialized_path', required=True, help='Path to specialized diffusion model')
    parser.add_argument('--save_aged_dir', default='./outputs', help='Path to save outputs')
    parser.add_argument('--target_ages', nargs='+', default=[10, 20, 40, 60, 80], type=int, help='Target age values')
    parser.add_argument("--seed", type=int, default=42, help="seed for seed_everything")

    args = parser.parse_args()
    
    return args

def classify_age(age_init):
    if age_init < 15:
        age_grouped = "child"
    elif age_init < 30:
        age_grouped = "teenager"
    elif age_init < 40:
        age_grouped = "youngadults"
    elif age_init < 50:
        age_grouped = "middleaged"
    elif age_init < 65:
        age_grouped = "elderly"
    else:
        age_grouped = "old"
    
    return age_grouped

def load_diffusers(specialized_path):
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                            clip_sample=False, set_alpha_to_one=False,
                            steps_offset=1)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    generator = torch.Generator(device=device)

    model = StableDiffusionPipeline.from_pretrained(specialized_path,
        scheduler=scheduler,
        safety_checker=None).to(device)
    tokenizer = model.tokenizer
    
    return model, generator, tokenizer

def invert(model, image_path, inversion_prompt):
    null_inversion = NullInversion(model)
    
    (_, _), x_t, uncond_embeddings = null_inversion.invert(
        image_path, inversion_prompt, offsets=(0,0,0,0), verbose=True)
    
    return x_t, uncond_embeddings

def prompt_to_prompt(inversion_prompt, prompts, model, generator, tokenizer, x_t, uncond_embeddings):
    prompt_before, prompt_after = prompts
    new_prompt = inversion_prompt
    for i in range(len(prompt_before)):
        new_prompt = new_prompt.replace(prompt_before[i], prompt_after[i])
    blend_word = ((prompt_before, prompt_after))
    is_replace_controller = True

    prompts = [inversion_prompt, new_prompt]

    cross_replace_steps = {'default_': .8,}
    self_replace_steps = .5

    eq_params = {"words": (prompt_after), "values": (1,)}

    controller = make_controller(prompts, is_replace_controller, cross_replace_steps, self_replace_steps,
                                tokenizer, blend_word, eq_params)

    images, _ = p2p_text2image(model, prompts, controller, generator=generator.manual_seed(0),
                            latent=x_t, uncond_embeddings=uncond_embeddings)
    return images

def save_output(images, save_aged_dir, output_img_name):
    new_img = images[-1]
    new_img_pil = Image.fromarray(new_img)
    new_img_pil.save(os.path.join(save_aged_dir, output_img_name))

if __name__ == '__main__':
	run()