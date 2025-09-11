from .datasets_new import VG_Relation, VG_Attribution, COCO_Order, Flickr_Order, COCO_Spatial, GQA_Spatial, Whatsup, Sugarcrepe, VL_CheckList
from .constants import VG_ARO_ROOT, COCO_ORDER_ROOT, FLICKR_ORDER_ROOT, COCO_SPATIAL_ROOT, GQA_SPATIAL_ROOT, WHATSUP_ROOT, SUGARCREPE_ROOT, VL_CHECKLIST_ROOT


def get_dataset(dataset_name, image_preprocess=None, text_perturb_fn=None, image_perturb_fn=None, download=False, *args, **kwargs):
    """
    Helper function that returns a dataset object with an evaluation function. 
    dataset_name: Name of the dataset.
    image_preprocess: Preprocessing function for images.
    text_perturb_fn: A function that takes in a string and returns a string. This is for perturbation experiments.
    image_perturb_fn: A function that takes in a PIL image and returns a PIL image. This is for perturbation experiments.
    download: Whether to allow downloading images if they are not found.
    """
    if dataset_name == "VG_Relation": 
        from .aro_datasets import get_visual_genome_relation
        return get_visual_genome_relation(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "VG_Attribution":
        from .aro_datasets import get_visual_genome_attribution
        return get_visual_genome_attribution(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "COCO_Order":
        from .aro_datasets import get_coco_order
        return get_coco_order(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "Flickr30k_Order":
        from .aro_datasets import get_flickr30k_order
        return get_flickr30k_order(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "Controlled_Images_A":
        from .aro_datasets import get_controlled_images_a
        return get_controlled_images_a(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "Controlled_Images_B":
        from .aro_datasets import get_controlled_images_b
        return get_controlled_images_b(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "COCO_QA_one_obj":
        from .aro_datasets import get_coco_qa_one_obj
        return get_coco_qa_one_obj(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "COCO_QA_two_obj":
        from .aro_datasets import get_coco_qa_two_obj
        return get_coco_qa_two_obj(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "VG_QA_one_obj":
        from .aro_datasets import get_vg_qa_one_obj
        return get_vg_qa_one_obj(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "VG_QA_two_obj":
        from .aro_datasets import get_vg_qa_two_obj
        return get_vg_qa_two_obj(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'VL_CheckList':
        from .aro_datasets import get_vl_checklist
        return get_vl_checklist(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "Sugarcrepe":
        from .aro_datasets import get_sugarcrepe
        return get_sugarcrepe(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "COCO_Retrieval":
        from .retrieval import get_coco_retrieval
        return get_coco_retrieval(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "Flickr30k_Retrieval":
        from .retrieval import get_flickr30k_retrieval
        return get_flickr30k_retrieval(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    

def get_dataset_new(dataset_name, image_preprocess=None, **kwargs):
    if dataset_name == "VG_Relation":
        return VG_Relation(
            root_dir=VG_ARO_ROOT,
            annotation_file='vg_relation',
            image_preprocess=image_preprocess,
            **kwargs
        )
    elif dataset_name == "VG_Attribution":
        return VG_Attribution(
            root_dir=VG_ARO_ROOT,
            annotation_file='vg_attribution',
            image_preprocess=image_preprocess,
            **kwargs
        )
    elif dataset_name == "COCO_Order":
        return COCO_Order(
            root_dir=COCO_ORDER_ROOT,
            image_preprocess=image_preprocess,
            **kwargs
        )
    elif dataset_name == "Flickr_Order":
        return Flickr_Order(
            root_dir=FLICKR_ORDER_ROOT,
            image_preprocess=image_preprocess,
            **kwargs
        )
    elif dataset_name == "COCO_Spatial_One":
        return COCO_Spatial(
            root_dir=COCO_SPATIAL_ROOT,
            annotation_file='coco_spatial_one',
            image_preprocess=image_preprocess,
            **kwargs
        )
    elif dataset_name == "COCO_Spatial_Two":
        return COCO_Spatial(
            root_dir=COCO_SPATIAL_ROOT,
            annotation_file='coco_spatial_two',
            image_preprocess=image_preprocess,
            **kwargs
        )
    elif dataset_name == "GQA_Spatial_One":
        return GQA_Spatial(
            root_dir=GQA_SPATIAL_ROOT,
            annotation_file='gqa_spatial_one',
            image_preprocess=image_preprocess,
            **kwargs
        )
    elif dataset_name == "GQA_Spatial_Two":
        return GQA_Spatial(
            root_dir=GQA_SPATIAL_ROOT,
            annotation_file='gqa_spatial_two',
            image_preprocess=image_preprocess,
            **kwargs
        )
    elif dataset_name == "Whatsup_A":
        return Whatsup(
            root_dir=WHATSUP_ROOT,
            annotation_file='whatsup_a',
            image_preprocess=image_preprocess,
            **kwargs
        )
    elif dataset_name == "Whatsup_B":
        return Whatsup(
            root_dir=WHATSUP_ROOT,
            annotation_file='whatsup_b',
            image_preprocess=image_preprocess,
            **kwargs
        )
    elif dataset_name == "Sugarcrepe":
        return Sugarcrepe(
            root_dir=SUGARCREPE_ROOT,
            image_preprocess=image_preprocess,
            **kwargs
        )
    elif dataset_name == "VL_CheckList":
        return VL_CheckList(
            root_dir=VL_CHECKLIST_ROOT,
            image_preprocess=image_preprocess,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
