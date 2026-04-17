import os
import sys
import numpy as np

import config as config
from bvp import solve_bvp_by_least_squares
from lsm_core import weighted_mse
from nonlinear_models import fit_all_models
from one_dimensional import (
    edit_weights_from_cli,
    fit_1d_polynomial,
    generate_1d_table,
    predict_1d_polynomial,
    slope_of_line,
)
from plotting import (
    plot_1d_fits,
    plot_2d_fits,
    plot_bvp_comparison,
    plot_model_selection,
    plot_weight_impact,
)
from report import save_1d_table_csv, save_2d_table_csv, save_text_report
from two_dimensional import fit_2d_polynomial, generate_2d_table, predict_2d_polynomial


UI_COLORS = {
    "reset": "\033[0m",
    "title": "\033[95m",
    "menu": "\033[35m",
    "accent": "\033[96m",
    "ok": "\033[92m",
    "warn": "\033[93m",
}


def enable_ansi_colors() -> None:
    if os.name == "nt":
        os.system("")


def ui_text(text: str, color_key: str) -> str:
    color = UI_COLORS.get(color_key, "")
    reset = UI_COLORS["reset"]

    return f"{color}{text}{reset}"


def build_one_d_table() -> dict:
    return generate_1d_table(
        nodes_count=config.ONE_D_CONFIG["nodes_count"],
        x_min=config.ONE_D_CONFIG["x_min"],
        x_max=config.ONE_D_CONFIG["x_max"],
        noise_sigma=config.ONE_D_CONFIG["noise_sigma"],
        random_seed=int(np.random.randint(0, 100000)),
    )


def init_session_state() -> dict:
    one_d_table = build_one_d_table()
    return {
        "one_d_table": one_d_table,
        "one_d_weights": one_d_table["weights"].copy(),
        "table_generation_count": 1,
    }


def print_title() -> None:
    line = "=" * 72
    print("\n" + ui_text(line, "title"))
    print(ui_text("ЛАБОРАТОРНАЯ РАБОТА №3. МЕТОД НАИМЕНЬШИХ КВАДРАТОВ", "title"))
    print(ui_text(line, "title"))


def print_menu() -> None:
    print("\n" + ui_text("-" * 72, "menu"))
    print(ui_text("1. Показать текущую 1D таблицу и веса", "menu"))
    print(ui_text("2. Изменить веса 1D таблицы", "menu"))
    print(ui_text("3. Сгенерировать новую 1D таблицу", "menu"))
    print(ui_text("4. Выполнить пункт 1 (1D аппроксимация)", "menu"))
    print(ui_text("5. Выполнить демонстрацию влияния весов", "menu"))
    print(ui_text("6. Выполнить пункт 2 (2D аппроксимация)", "menu"))
    print(ui_text("7. Выполнить пункт 3 (выбор лучшей модели)", "menu"))
    print(ui_text("8. Выполнить пункт 4 (краевая задача)", "menu"))
    print(ui_text("9. Выполнить все пункты и итоговый отчет", "menu"))
    print(ui_text("0. Выход", "warn"))
    print(ui_text("-" * 72, "menu"))


def print_one_d_table(table: dict, weights: np.ndarray) -> None:
    print("\n" + ui_text("Текущая 1D таблица:", "accent"))
    print(ui_text(" i |        x |         y |      rho", "accent"))
    print(ui_text("---+----------+-----------+----------", "accent"))
    for index, (x_value, y_value, weight_value) in enumerate(zip(table["x"], table["y"], weights)):
        print(f"{index:>2} | {x_value:>8.4f} | {y_value:>9.4f} | {weight_value:>8.4f}")


def run_one_dimensional_part(results_dir: str, one_d_table: dict, user_weights: np.ndarray) -> dict:

    fit_n1 = fit_1d_polynomial(one_d_table["x"], one_d_table["y"], user_weights, degree=1)
    fit_n2 = fit_1d_polynomial(one_d_table["x"], one_d_table["y"], user_weights, degree=2)
    fit_n3 = fit_1d_polynomial(one_d_table["x"], one_d_table["y"], user_weights, degree=3)

    predicted_n1 = predict_1d_polynomial(one_d_table["x"], fit_n1)
    predicted_n2 = predict_1d_polynomial(one_d_table["x"], fit_n2)
    predicted_n3 = predict_1d_polynomial(one_d_table["x"], fit_n3)

    mse_n1 = weighted_mse(one_d_table["y"], predicted_n1, user_weights)
    mse_n2 = weighted_mse(one_d_table["y"], predicted_n2, user_weights)
    mse_n3 = weighted_mse(one_d_table["y"], predicted_n3, user_weights)

    plot_1d_fits(
        one_d_table,
        [fit_n1, fit_n2, fit_n3],
        os.path.join(results_dir, "part1_1d_fits.png"),
    )

    save_1d_table_csv(
        os.path.join(results_dir, "part1_table.csv"),
        one_d_table["x"],
        one_d_table["y"],
        user_weights,
    )

    return {
        "table": one_d_table,
        "weights": user_weights,
        "fit_n1": fit_n1,
        "fit_n2": fit_n2,
        "fit_n3": fit_n3,
        "mse_n1": mse_n1,
        "mse_n2": mse_n2,
        "mse_n3": mse_n3,
    }


def run_two_dimensional_part(results_dir: str, show_interactive: bool = False) -> dict:
    two_d_table = generate_2d_table(
        nodes_count=config.TWO_D_CONFIG["nodes_count"],
        x_min=config.TWO_D_CONFIG["x_min"],
        x_max=config.TWO_D_CONFIG["x_max"],
        y_min=config.TWO_D_CONFIG["y_min"],
        y_max=config.TWO_D_CONFIG["y_max"],
        noise_sigma=config.TWO_D_CONFIG["noise_sigma"],
        random_seed=config.RANDOM_SEED + 11,
    )

    fit_deg1 = fit_2d_polynomial(
        two_d_table["x"],
        two_d_table["y"],
        two_d_table["z"],
        two_d_table["weights"],
        degree=1,
    )
    fit_deg2 = fit_2d_polynomial(
        two_d_table["x"],
        two_d_table["y"],
        two_d_table["z"],
        two_d_table["weights"],
        degree=2,
    )

    z_pred_deg1 = predict_2d_polynomial(two_d_table["x"], two_d_table["y"], fit_deg1)
    z_pred_deg2 = predict_2d_polynomial(two_d_table["x"], two_d_table["y"], fit_deg2)

    mse_deg1 = weighted_mse(two_d_table["z"], z_pred_deg1, two_d_table["weights"])
    mse_deg2 = weighted_mse(two_d_table["z"], z_pred_deg2, two_d_table["weights"])

    plot_2d_fits(
        two_d_table,
        fit_deg1,
        fit_deg2,
        os.path.join(results_dir, "part2_2d_fits.png"),
        show_interactive=show_interactive,
    )

    save_2d_table_csv(
        os.path.join(results_dir, "part2_table.csv"),
        two_d_table["x"],
        two_d_table["y"],
        two_d_table["z"],
        two_d_table["weights"],
    )

    return {
        "fit_deg1": fit_deg1,
        "fit_deg2": fit_deg2,
        "mse_deg1": mse_deg1,
        "mse_deg2": mse_deg2,
    }


def run_model_selection_part(results_dir: str) -> dict:
    x = np.array(config.MODEL_SELECTION_DATA["x"], dtype=float)
    y = np.array(config.MODEL_SELECTION_DATA["y"], dtype=float)
    weights = np.array(config.MODEL_SELECTION_DATA["weights"], dtype=float)

    models = fit_all_models(x, y, weights)

    plot_model_selection(
        x,
        y,
        models,
        os.path.join(results_dir, "part3_model_selection.png"),
    )

    return {
        "models": models,
        "best_model": models[0],
    }


def run_bvp_part(results_dir: str) -> dict:
    solution_m2 = solve_bvp_by_least_squares(2, grid_points=config.BVP_CONFIG["grid_points"])
    solution_m3 = solve_bvp_by_least_squares(3, grid_points=config.BVP_CONFIG["grid_points"])

    plot_bvp_comparison(
        solution_m2,
        solution_m3,
        os.path.join(results_dir, "part4_bvp_comparison.png"),
    )

    return {
        "solution_m2": solution_m2,
        "solution_m3": solution_m3,
    }


def make_summary_lines(part1: dict, impact_demo: dict, part2: dict, part3: dict, part4: dict) -> list[str]:
    lines = []

    lines.append("ЛР3. Метод наименьших квадратов")
    lines.append("=" * 45)

    lines.append("\n1) Одномерная аппроксимация")
    lines.append(f"MSE n=1: {part1['mse_n1']:.8f}")
    lines.append(f"MSE n=2: {part1['mse_n2']:.8f}")
    lines.append(f"MSE n=3: {part1['mse_n3']:.8f}")

    lines.append("\n1b) Влияние весов на прямую n=1")
    lines.append(f"k (rho=1): {impact_demo['slope_uniform']:.8f}")
    lines.append(f"k (назначенные веса): {impact_demo['slope_custom']:.8f}")

    lines.append("\n2) Двумерная аппроксимация")
    lines.append(f"MSE degree=1: {part2['mse_deg1']:.8f}")
    lines.append(f"MSE degree=2: {part2['mse_deg2']:.8f}")

    lines.append("\n3) Выбор лучшей нелинейной модели")
    for model in part3["models"]:
        if np.isfinite(model["mse"]):
            lines.append(f"{model['name']}: mse={model['mse']:.8f}, params={model['params']}")
        else:
            lines.append(f"{model['name']}: недоступна ({model.get('error', 'ошибка')})")
    lines.append(f"Лучшая модель: {part3['best_model']['name']}")

    lines.append("\n4) Краевая задача ODE")
    lines.append("Использован оператор: y'' + x*y' + y = x")
    lines.append(f"m=2, ||res||_L2 = {part4['solution_m2']['residual_l2']:.8f}")
    lines.append(f"m=3, ||res||_L2 = {part4['solution_m3']['residual_l2']:.8f}")

    return lines


def run_all_parts(results_dir: str, state: dict) -> None:
    from one_dimensional import create_inverted_weights
    
    part1 = run_one_dimensional_part(results_dir, state["one_d_table"], state["one_d_weights"])
    
    current_x = state["one_d_table"]["x"]
    current_y = state["one_d_table"]["y"]
    uniform_weights = np.ones_like(current_x)
    inverted_weights = create_inverted_weights(uniform_weights, first_half_weight=6.0)
    fit_uniform = fit_1d_polynomial(current_x, current_y, uniform_weights, degree=1)
    fit_inverted = fit_1d_polynomial(current_x, current_y, inverted_weights, degree=1)
    impact_demo = {
        "slope_uniform": slope_of_line(fit_uniform),
        "slope_custom": slope_of_line(fit_inverted),
    }
    # Передаем веса для визуализации
    plot_weight_impact(
        {"x": current_x, "y": current_y, "weights": inverted_weights},
        fit_uniform,
        fit_inverted,
        os.path.join(results_dir, "part1_weight_impact.png"),
    )
    
    part2 = run_two_dimensional_part(results_dir)
    part3 = run_model_selection_part(results_dir)
    part4 = run_bvp_part(results_dir)

    summary_lines = make_summary_lines(part1, impact_demo, part2, part3, part4)
    save_text_report(os.path.join(results_dir, "summary.txt"), summary_lines)

    print("\n" + ui_text("✓ Все пункты выполнены. Отчет: results/summary.txt", "ok"))


def process_menu_choice(choice: str, results_dir: str, state: dict) -> bool:
    if choice == "1":
        print_one_d_table(state["one_d_table"], state["one_d_weights"])
        return True

    if choice == "2":
        state["one_d_weights"] = edit_weights_from_cli(state["one_d_weights"])
        # Обновляем веса и в таблице
        state["one_d_table"]["weights"] = state["one_d_weights"].copy()
        print(ui_text("✓ Веса обновлены.", "ok"))
        return True

    if choice == "3":
        state["one_d_table"] = build_one_d_table()
        state["one_d_weights"] = state["one_d_table"]["weights"].copy()
        state["table_generation_count"] = state.get("table_generation_count", 0) + 1
        save_1d_table_csv(
            os.path.join(results_dir, "part1_table.csv"),
            state["one_d_table"]["x"],
            state["one_d_table"]["y"],
            state["one_d_weights"],
        )
        print(ui_text(f"✓ Новая таблица #{state['table_generation_count']} создана, веса сброшены.", "ok"))
        return True

    if choice == "4":
        # Синхронизируем веса в таблице перед вычислениями и сохранением
        state["one_d_table"]["weights"] = state["one_d_weights"].copy()
        part1 = run_one_dimensional_part(results_dir, state["one_d_table"], state["one_d_weights"])
        print("\n" + ui_text("✓ Пункт 1 (1D аппроксимация) готов", "ok"))
        print(f"MSE n=1: {part1['mse_n1']:.8f}")
        print(f"MSE n=2: {part1['mse_n2']:.8f}")
        print(f"MSE n=3: {part1['mse_n3']:.8f}")
        return True

    if choice == "5":
        from one_dimensional import create_inverted_weights

        current_x = state["one_d_table"]["x"]
        current_y = state["one_d_table"]["y"]
        uniform_weights = np.ones_like(current_x)
        inverted_weights = create_inverted_weights(uniform_weights, first_half_weight=6.0)

        fit_uniform = fit_1d_polynomial(current_x, current_y, uniform_weights, degree=1)
        fit_inverted = fit_1d_polynomial(current_x, current_y, inverted_weights, degree=1)

        slope_uniform = slope_of_line(fit_uniform)
        slope_inverted = slope_of_line(fit_inverted)

        # Передаем веса для визуализации
        plot_weight_impact(
            {"x": current_x, "y": current_y, "weights": inverted_weights},
            fit_uniform,
            fit_inverted,
            os.path.join(results_dir, "part1_weight_impact.png"),
        )

        print("\n" + ui_text("✓ Демонстрация влияния весов готова", "ok"))
        print(f"k (rho=1 для всех): {slope_uniform:.8f}")
        print(f"k (противоположные веса): {slope_inverted:.8f}")
        sign_change = "✓ ДА - знак изменился!" if slope_uniform * slope_inverted < 0 else "нет, одинаковый знак"
        print(f"Изменение знака наклона: {sign_change}")
        return True

    if choice == "6":
        print("\n" + ui_text("Режим 2D графика: 1=файл, 2=интерактив", "accent"))
        view_mode = input(ui_text("Выбор (1/2)? ", "accent")).strip()

        show_interactive = view_mode == "2"
        part2 = run_two_dimensional_part(results_dir, show_interactive=show_interactive)
        print("\n" + ui_text("✓ Пункт 2 (2D аппроксимация) готов", "ok"))
        print(f"MSE degree=1: {part2['mse_deg1']:.8f}")
        print(f"MSE degree=2: {part2['mse_deg2']:.8f}")
        return True

    if choice == "7":
        part3 = run_model_selection_part(results_dir)
        print("\n" + ui_text("✓ Пункт 3 (выбор модели) готов", "ok"))
        print(f"Лучшая модель: {part3['best_model']['name']}")
        for model in part3["models"]:
            if np.isfinite(model["mse"]):
                print(f"  {model['name']}: mse={model['mse']:.8f}")
        return True

    if choice == "8":
        part4 = run_bvp_part(results_dir)
        print("\n" + ui_text("✓ Пункт 4 (краевая задача) готов", "ok"))
        print(f"m=2, ||res||_L2 = {part4['solution_m2']['residual_l2']:.8f}")
        print(f"m=3, ||res||_L2 = {part4['solution_m3']['residual_l2']:.8f}")
        return True

    if choice == "9":
        run_all_parts(results_dir, state)
        return True

    if choice == "0":
        print(ui_text("До встречи!", "warn"))
        return False

    print(ui_text("⚠ Неизвестный пункт. Попробуйте снова.", "warn"))
    return True


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    state = init_session_state()

    if not sys.stdin.isatty():
        run_all_parts(results_dir, state)
        return

    enable_ansi_colors()
    print_title()
    should_continue = True

    while should_continue:
        print_menu()
        user_choice = input("\n" + ui_text("Введите номер пункта: ", "accent")).strip()
        should_continue = process_menu_choice(user_choice, results_dir, state)


if __name__ == "__main__":
    main()
