import os
import json


def generate_list_json(root_dir):
    """
    遍历 root_dir 下的所有子文件夹，
    在每个包含图片的文件夹里生成 list.json
    每个元素格式：{"file_name": "图片完整名称", "caption": ""}
    """
    # 支持的图片后缀
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

    folder_count = 0
    result_folders = []  # 最后返回的列表，有多少个图片文件夹就有多少个字符串
    captions = [
        "vqz monochrome grayscale image of a railway bogie foundation brake assembly, extremely dark and low-key, most of the frame sunk in near-black shadow; a heavy forged steel brake lever with a curved S-shaped silhouette runs diagonally through the center of the frame, its matte cast-metal surface worn with chipped paint, fine scratches and faint rust stains; large hexagonal bolts with protruding threaded studs fasten the lever at its upper and lower joints; embossed cast numbers and letters are barely legible along the lever's top face; to the right of the lever, a rounded cast metal housing sits above a ribbed corrugated rubber boot enclosing a coiled spring, all rendered in dim mid-gray tones; a second horizontal steel link with a hexagonal bolt crosses the lower part of the frame; along the left edge, dark curved concentric bands of a wheel or brake disc fade into blackness; a single weak light from above grazes the metal edges with soft faint highlights while everything behind dissolves into pure black; the whole image is fully desaturated to gray, heavily underexposed and covered with fine grain and sensor noise, giving it a dim infrared night-vision look.",
        "xqf monochrome grayscale image of a railway bogie underframe corner, extremely dark and low-key with most of the frame sunk in near-black shadow; a broad flat cast steel plate with a speckled dusty surface fills the left and upper part of the frame, its edge curving down toward the center; a triangular gusset plate with a round hole stands along the left edge in deep shadow; a twisted steel wire rope loop arcs across the middle of the frame, its two ends crimped and bolted to small brackets on the plate; at the center-right an upright cast bracket carries a large hexagonal bolt secured with a thin split pin; beside it a round flanged hub cover with concentric ridges faces the viewer; a heavy inclined steel plate with two round bolt holes runs down the right side of the frame; in the lower right corner a ribbed cast housing with hexagonal bolts emerges from the darkness; below the plate further bolted brackets and threaded studs fade into black; a single weak light from above grazes the metal edges and the wire rope strands with soft faint highlights while the background dissolves into pure black; the whole image is fully desaturated to gray, heavily underexposed and covered with fine grain and sensor noise, giving it a dim infrared night-vision look.",
        "kqz monochrome grayscale image of a railway underfloor equipment manifold, extremely dark and low-key with most of the frame sunk in near-black shadow; a long rectangular cast metal block runs diagonally from the top center to the middle right of the frame, its flat top face stamped with embossed letters and marks; three hexagonal gland nuts sit in a row along the block's lower face, each feeding a short stub into a small rectangular clamp box topped with two hexagonal bolts, the three boxes stepping down in a diagonal line; from each clamp box a spiral-wrapped ribbed hose sleeve extends in parallel diagonals toward the lower left corner, their coiled guards catching faint highlights; a large smooth cylindrical body fills the upper right corner in pale gray; along the right edge a thin cable and a small bolted bracket plate fade into the darkness; across the bottom of the frame two smooth dark hoses with ringed metal collars bend toward the lower right corner, much of their length lost in shadow as they emerge from and sink back into the blackness; a single weak light from above grazes the cast edges and bolt heads with soft faint highlights while the background dissolves into pure black; the whole image is fully desaturated to gray, heavily underexposed and covered with fine grain and sensor noise, giving it a dim infrared night-vision look.",
        "wqz monochrome grayscale image of a cast steel housing corner on a railway bogie, extremely dark and low-key with broad areas sunk in deep shadow; across the top of the frame the bolted rim of a large circular cover sweeps in a wide arc, its hexagonal bolt heads spaced along the curved flange and catching small bright glints; at the left of center a raised rectangular pad tilts diagonally, its upper recess holding an oval bright metal plate fastened by four hexagonal bolts with a narrow slotted opening through the middle, and below it a large hexagonal plug seated on a square flange ringed by four socket-head cap screws, a twisted lockwire looped through the plug and trailing to the side; at the right of center a round raised boss carries a four-bolt flange with a large central hexagonal bolt, its thin twisted safety wire curling down across the boss face; the surrounding cast housing wears glossy dark paint, its rippled surface catching broad soft highlights along the lower right; deep black shadows swallow the gaps between the pads and dissolve the background; the whole image is fully desaturated to gray, heavily underexposed and covered with fine grain and sensor noise, giving it a dim infrared night-vision look."
    ]
    for index, name in enumerate(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, name)
        if not os.path.isdir(folder_path):
            continue

        images = []
        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_exts:
                images.append({
                    "file_name": file,  # 完整文件名（含后缀）
                    "caption": captions[index]
                })

        if images:  # 只有存在图片的文件夹才生成 json
            json_path = os.path.join(folder_path, "list.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(images, f, ensure_ascii=False, indent=2)

            print(f"已生成: {json_path}  (共 {len(images)} 张图片)")
            folder_count += 1
            result_folders.append("")  # 预留一个空字符串，方便你后面填 caption

    print(f"\n总共处理了 {folder_count} 个图片文件夹")
    # return result_folders


if __name__ == "__main__":
    # ========== 修改这里 ==========
    root_directory = r"F:\dataset\flux_2_test"  # 改成你实际的文件夹路径
    # =============================

    caption_list = generate_list_json(root_directory)
    #
    # # 最终返回的列表（有多少个图片文件夹就有多少个字符串）
    # print("\n最终列表（可直接用来填 caption）:")
    # print(caption_list)