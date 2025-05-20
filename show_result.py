import pandas as pd
import argparse

# VL_CheckList Results
def get_vl_checklist_results(output_dir):
    csv_file = f'{output_dir}/VL_CheckList.csv'
    df = pd.read_csv(csv_file)

    vl_res = []
    for index, row in df.iterrows():
        vl_res.append(['VL_CheckList', row['Attributes'], row['Accuracy'] * 100])

    return vl_res

def get_sugarcrepe_results(output_dir):
    csv_file = f'{output_dir}/Sugarcrepe.csv'
    df = pd.read_csv(csv_file)

    sugarcrepe_res = []
    sugarcrepe_row = [
        'add_att', 'replace_att', 'swap_att',
        'add_obj', 'replace_obj', 'swap_obj',
        'replace_rel',
        'Overall'
    ]
    
    for item in sugarcrepe_row:
        for index, row in df.iterrows():
            if row['Attributes'] == item:
                sugarcrepe_res.append(['Sugarcrepe', row['Attributes'], row['Accuracy'] * 100])
                break

    return sugarcrepe_res

def get_aro_results(output_dir):
    aro_datasets = [
        "VG_Attribution",
        "VG_Relation",
        "COCO_Order",
        "Flickr30k_Order",
    ]

    aro_res = []
    sum_acc = 0
    for dataset in aro_datasets:
        # Load the CSV file
        csv_file = f'{output_dir}/{dataset}.csv'
        df = pd.read_csv(csv_file)
        acc = float(df['Accuracy'].iloc[-1] * 100)
        aro_res.append(['ARO', dataset, acc])
        sum_acc += acc
    aro_res.append(['ARO', 'Overall', sum_acc / len(aro_datasets)])
    return aro_res

def get_whatsup_results(output_dir):
    whatsup_datasets = [
        "Controlled_Images_A",
        "Controlled_Images_B",
    ]

    whatsup_res = []
    sum_acc = 0
    for dataset in whatsup_datasets:
        # Load the CSV file
        csv_file = f'{output_dir}/{dataset}.csv'
        df = pd.read_csv(csv_file)
        acc = float(df['Accuracy'].iloc[-1] * 100)
        whatsup_res.append(['Whatsup', dataset, acc])
        sum_acc += acc
    whatsup_res.append(['Whatsup', 'Overall', sum_acc / len(whatsup_datasets)])
    return whatsup_res

def get_coco_spatial_results(output_dir):
    coco_spatial_datasets = [
        "COCO_QA_one_obj",
        "COCO_QA_two_obj",
    ]

    coco_spatial_res = []
    sum_acc = 0
    for dataset in coco_spatial_datasets:
        # Load the CSV file
        csv_file = f'{output_dir}/{dataset}.csv'
        df = pd.read_csv(csv_file)
        acc = float(df['Accuracy'].iloc[-1] * 100)
        coco_spatial_res.append(['COCO-Spatial', dataset, acc])
        sum_acc += acc
    coco_spatial_res.append(['COCO-Spatial', 'Overall', sum_acc / len(coco_spatial_datasets)])
    return coco_spatial_res

def get_gqa_spatial_results(output_dir):
    gqa_spatial_datasets = [
        "VG_QA_one_obj",
        "VG_QA_two_obj",
    ]

    gqa_spatial_res = []
    sum_acc = 0
    for dataset in gqa_spatial_datasets:
        # Load the CSV file
        csv_file = f'{output_dir}/{dataset}.csv'
        df = pd.read_csv(csv_file)
        acc = float(df['Accuracy'].iloc[-1] * 100)
        gqa_spatial_res.append(['GQA-Spatial', dataset, acc])
        sum_acc += acc
    gqa_spatial_res.append(['GQA-Spatial', 'Overall', sum_acc / len(gqa_spatial_datasets)])
    return gqa_spatial_res

def get_all_results(output_dir):
    all_results = []
    all_results.extend(get_vl_checklist_results(output_dir))
    all_results.extend(get_sugarcrepe_results(output_dir))
    all_results.extend(get_aro_results(output_dir))
    all_results.extend(get_coco_spatial_results(output_dir))
    all_results.extend(get_gqa_spatial_results(output_dir))
    all_results.extend(get_whatsup_results(output_dir))


    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str)
    args = parser.parse_args()

    exp_name = args.exp_name
    output_dir = f'/mnt/user_data/yifei/outputs/whatsup/{exp_name}'

    all_results = get_all_results(output_dir)
    # for result in all_results:
    #     print(result)

    # 生成 Markdown 表格
    md_table = f"| {all_results[0][0]} | {all_results[0][1]} | {all_results[0][2]:.2f} |\n"
    md_table += "|----------|-------------|-------|\n"

    for row in all_results[1:]:
        md_table += f"| {row[0]} | {row[1]} | {row[2]:.2f} |\n"
    print(md_table)
