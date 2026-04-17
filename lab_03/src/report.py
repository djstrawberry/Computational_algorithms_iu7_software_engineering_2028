import csv


def save_1d_table_csv(file_path: str, x, y, weights) -> None:
    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["i", "x", "y", "rho"])
        for index, (x_item, y_item, weight_item) in enumerate(zip(x, y, weights)):
            writer.writerow([index, f"{x_item:.8f}", f"{y_item:.8f}", f"{weight_item:.8f}"])


def save_2d_table_csv(file_path: str, x, y, z, weights) -> None:
    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["i", "x", "y", "z", "rho"])
        for index, (x_item, y_item, z_item, weight_item) in enumerate(zip(x, y, z, weights)):
            writer.writerow([index, f"{x_item:.8f}", f"{y_item:.8f}", f"{z_item:.8f}", f"{weight_item:.8f}"])


def save_text_report(file_path: str, report_lines: list[str]) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))
