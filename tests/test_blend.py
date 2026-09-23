"""POST /api/v1/solutions/blend 的 HTTP API 测试。

覆盖：单母液稀释、多母液唯一解、欠定多解时母液总量最小择优、同总量下按
输入顺序字典序择优、容量/终体积约束生效、补水为 0、母液专属溶质的终
浓度回显、无限小数精确渲染，以及 422 输入校验（母液过多、标识/目标
重复、未知溶质、非法十进制、JSON 数字、零终体积、空白字符串）与
BLEND_NO_SOLUTION（浓度矛盾、可用体积不足、总量超出终体积）。
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PATH = "/api/v1/solutions/blend"


def stock(sid, concentrations, volume="100"):
    return {
        "id": sid,
        "concentrations": concentrations,
        "available_volume_ml": volume,
    }


def body(stocks_, final_volume, targets):
    return {
        "stocks": stocks_,
        "final_volume_ml": final_volume,
        "targets": [{"solute": s, "concentration": c} for s, c in targets],
    }


def takes(data):
    return {item["id"]: item["taken_volume_ml"] for item in data["stocks"]}


# ------------------------------------------------------------ 成功路径
def test_single_stock_dilution():
    # 1 mol/L NaCl 取 25 mL 稀释到 250 mL -> 0.1 mol/L
    resp = client.post(
        PATH,
        json=body([stock("A", {"NaCl": "1"})], "250", [("NaCl", "0.1")]),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert takes(data) == {"A": "25"}
    assert data["total_stock_volume_ml"] == "25"
    assert data["water_volume_ml"] == "225"
    assert data["final_volume_ml"] == "250"
    assert data["stocks"][0]["available_volume_ml"] == "100"
    fc = data["final_concentrations"][0]
    assert fc == {
        "solute": "NaCl",
        "target_concentration": "0.1",
        "final_concentration": "0.1",
    }


def test_two_stocks_unique_solution():
    # 纯 NaCl 母液 + 纯 KCl 母液，两个独立目标 -> 唯一解
    resp = client.post(
        PATH,
        json=body(
            [stock("A", {"NaCl": "1", "KCl": "0"}),
             stock("B", {"NaCl": "0", "KCl": "1"})],
            "100",
            [("NaCl", "0.2"), ("KCl", "0.3")],
        ),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert takes(data) == {"A": "20", "B": "30"}
    assert data["total_stock_volume_ml"] == "50"
    assert data["water_volume_ml"] == "50"
    # 终浓度顺序遵循 targets 的首次出现顺序
    assert [item["solute"] for item in data["final_concentrations"]] == [
        "NaCl",
        "KCl",
    ]


def test_zero_targets_uses_water_only():
    # 所有目标浓度为 0 -> 不用任何母液，全部补水（总量最小）
    resp = client.post(
        PATH,
        json=body(
            [stock("A", {"NaCl": "1"}, "50"), stock("B", {"KCl": "1"}, "50")],
            "100",
            [("NaCl", "0"), ("KCl", "0")],
        ),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert takes(data) == {"A": "0", "B": "0"}
    assert data["total_stock_volume_ml"] == "0"
    assert data["water_volume_ml"] == "100"


def test_water_zero_when_target_equals_stock_concentration():
    # 目标浓度等于母液浓度且 V 不超过可用量 -> 全用母液、补水 0
    resp = client.post(
        PATH,
        json=body([stock("A", {"NaCl": "1"})], "50", [("NaCl", "1")]),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert takes(data) == {"A": "50"}
    assert data["water_volume_ml"] == "0"


def test_min_total_prefers_more_concentrated_stock():
    # 1 mol/L 母液（容量受限）与 0.5 mol/L 母液共同满足 50 mmol：
    # 先取满浓母液 A、B 各 20 mL，再用淡母液 20 mL，总量 60 mL 为最小；
    # 淡母液 C、D 等价，字典序最小把 0 放在靠前的 C -> (20, 20, 0, 20)
    resp = client.post(
        PATH,
        json=body(
            [
                stock("A", {"NaCl": "1"}, "20"),
                stock("B", {"NaCl": "1"}, "20"),
                stock("C", {"NaCl": "0.5"}, "60"),
                stock("D", {"NaCl": "0.5"}, "60"),
            ],
            "100",
            [("NaCl", "0.5")],
        ),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert takes(data) == {"A": "20", "B": "20", "C": "0", "D": "20"}
    assert data["total_stock_volume_ml"] == "60"
    assert data["water_volume_ml"] == "40"


def test_lexicographic_tiebreak_equal_concentrations():
    # 两种完全等价的 1 mol/L 母液：总量固定 10 mL 时，
    # 按输入顺序字典序最小 -> 第一种取 0、第二种取 10
    resp = client.post(
        PATH,
        json=body(
            [
                stock("A", {"NaCl": "1"}, "100"),
                stock("B", {"NaCl": "1"}, "100"),
            ],
            "100",
            [("NaCl", "0.1")],
        ),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert takes(data) == {"A": "0", "B": "10"}
    assert data["water_volume_ml"] == "90"


def test_lexicographic_order_follows_input_order():
    # 调换母液输入顺序后，字典序最优先把取用量压到靠前的母液
    resp = client.post(
        PATH,
        json=body(
            [
                stock("B", {"NaCl": "1"}, "100"),
                stock("A", {"NaCl": "1"}, "100"),
            ],
            "100",
            [("NaCl", "0.1")],
        ),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert [item["id"] for item in data["stocks"]] == ["B", "A"]
    assert takes(data) == {"B": "0", "A": "10"}


def test_lexicographic_tiebreak_with_two_free_dimensions():
    # xA+xB = 20（各容量 20），xC+xD = 20（各容量 50），总量固定 40；
    # 字典序最小依次把 A、C 压到 0 -> (0, 20, 0, 20)
    resp = client.post(
        PATH,
        json=body(
            [
                stock("A", {"NaCl": "1", "KCl": "0"}, "20"),
                stock("B", {"NaCl": "1", "KCl": "0"}, "20"),
                stock("C", {"NaCl": "0.5", "KCl": "1"}, "50"),
                stock("D", {"NaCl": "0.5", "KCl": "1"}, "50"),
            ],
            "100",
            [("NaCl", "0.3"), ("KCl", "0.2")],
        ),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert takes(data) == {"A": "0", "B": "20", "C": "0", "D": "20"}
    assert data["water_volume_ml"] == "60"
    finals = {item["solute"]: item["final_concentration"]
              for item in data["final_concentrations"]}
    assert finals == {"NaCl": "0.3", "KCl": "0.2"}


def test_stock_only_solute_reported_with_null_target():
    # 目标只约束 NaCl，但母液还含 KCl：终浓度须回显，target 为 null
    resp = client.post(
        PATH,
        json=body([stock("A", {"NaCl": "1", "KCl": "0.5"})], "250",
                  [("NaCl", "0.1")]),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert takes(data) == {"A": "25"}
    by_solute = {item["solute"]: item for item in data["final_concentrations"]}
    assert list(by_solute) == ["NaCl", "KCl"]  # 目标溶质在前，其余按名字升序
    assert by_solute["KCl"]["target_concentration"] is None
    assert by_solute["KCl"]["final_concentration"] == "0.05"
    assert by_solute["NaCl"]["target_concentration"] == "0.1"


def test_repeating_decimal_rendered_exactly_28_sig_digits():
    # 3 mol/L 母液配制 1 mol/L × 100 mL：取用量 100/3，无限小数按 28 位
    # 有效数字银行家舍入；补水 200/3 同理
    resp = client.post(
        PATH,
        json=body([stock("A", {"NaCl": "3"})], "100",
                  [("NaCl", "1")]),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert takes(data)["A"] == "33.33333333333333333333333333"
    assert data["water_volume_ml"] == "66.66666666666666666666666667"


def test_scientific_notation_inputs_accepted():
    resp = client.post(
        PATH,
        json=body([stock("A", {"NaCl": "1e0"}, "1e2")], "2.5e2",
                  [("NaCl", "1e-1")]),
    )
    assert resp.status_code == 200, resp.text
    assert takes(resp.json()) == {"A": "25"}


def test_zero_available_volume_stock_simply_unused():
    # 可用体积为 0 的母液可以提交，但不能被取用
    resp = client.post(
        PATH,
        json=body(
            [
                stock("EMPTY", {"NaCl": "10"}, "0"),
                stock("FULL", {"NaCl": "1"}, "100"),
            ],
            "100",
            [("NaCl", "0.1")],
        ),
    )
    assert resp.status_code == 200, resp.text
    assert takes(resp.json()) == {"EMPTY": "0", "FULL": "10"}


def test_zero_concentration_listing_is_redundant():
    # 母液以 "0" 列出某溶质、目标也为 0：等式 0=0 不影响求解
    resp = client.post(
        PATH,
        json=body(
            [stock("A", {"NaCl": "1", "KCl": "0"})], "100",
            [("NaCl", "0.1"), ("KCl", "0")],
        ),
    )
    assert resp.status_code == 200, resp.text
    assert takes(resp.json()) == {"A": "10"}


def test_same_solute_name_across_stocks_merges():
    # 两种母液都含 NaCl，浓度叠加满足目标
    resp = client.post(
        PATH,
        json=body(
            [
                stock("A", {"NaCl": "0.2"}, "100"),
                stock("B", {"NaCl": "0.1"}, "100"),
            ],
            "100",
            [("NaCl", "0.12")],
        ),
    )
    # 0.2 xA + 0.1 xB = 12，min(xA+xB) 尽量多用浓母液 A：xA=60, xB=0
    assert resp.status_code == 200, resp.text
    assert takes(resp.json()) == {"A": "60", "B": "0"}


# ------------------------------------------------------------ 无解
def test_no_solution_when_required_volume_exceeds_capacity():
    # 目标 1 mol/L × 500 mL 需 500 mmol，母液只有 100 mL × 1 mol/L
    resp = client.post(
        PATH,
        json=body([stock("A", {"NaCl": "1"}, "100")], "500",
                  [("NaCl", "1")]),
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "BLEND_NO_SOLUTION"
    assert err["details"]["stock_ids"] == ["A"]
    assert err["details"]["targets"] == [
        {"solute": "NaCl", "concentration": "1"}
    ]
    assert err["details"]["final_volume_ml"] == "500"


def test_no_solution_when_target_more_concentrated_than_any_stock():
    # 0.5 mol/L 母液无法配制 1 mol/L 终溶液：所需体积超过终体积
    resp = client.post(
        PATH,
        json=body([stock("A", {"NaCl": "0.5"}, "100")], "50",
                  [("NaCl", "1")]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "BLEND_NO_SOLUTION"


def test_no_solution_when_solute_missing_from_all_stocks_positive():
    resp = client.post(
        PATH,
        json=body([stock("A", {"NaCl": "1"}, "100")], "100",
                  [("KCl", "0.1")]),
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "INVALID_BLEND"
    assert err["details"]["unknown_solutes"] == ["KCl"]


def test_no_solution_zero_target_unknown_solute_still_invalid():
    # 即使目标浓度为 0，溶质也必须在某种母液中出现
    resp = client.post(
        PATH,
        json=body([stock("A", {"NaCl": "1"}, "100")], "100",
                  [("KCl", "0")]),
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "INVALID_BLEND"
    assert err["details"]["unknown_solutes"] == ["KCl"]


def test_no_solution_inconsistent_targets():
    # 两种母液都是 NaCl/KCl 等浓度，无法让终溶液 NaCl 与 KCl 浓度不同
    resp = client.post(
        PATH,
        json=body(
            [
                stock("A", {"NaCl": "1", "KCl": "1"}, "100"),
                stock("B", {"NaCl": "0.5", "KCl": "0.5"}, "100"),
            ],
            "100",
            [("NaCl", "0.3"), ("KCl", "0.2")],
        ),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "BLEND_NO_SOLUTION"


def test_no_solution_zero_volume_only_stock_cannot_supply():
    resp = client.post(
        PATH,
        json=body([stock("A", {"NaCl": "1"}, "0")], "100",
                  [("NaCl", "0.1")]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "BLEND_NO_SOLUTION"


# ------------------------------------------------------------ 422 输入校验
def test_too_many_stocks_rejected():
    resp = client.post(
        PATH,
        json=body(
            [stock(sid, {"NaCl": "1"}) for sid in ("A", "B", "C", "D", "E")],
            "100",
            [("NaCl", "0.1")],
        ),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_duplicate_stock_ids_rejected():
    resp = client.post(
        PATH,
        json=body(
            [stock("A", {"NaCl": "1"}), stock("A", {"KCl": "1"})],
            "100",
            [("NaCl", "0.1")],
        ),
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "INVALID_BLEND"
    assert err["details"]["duplicate_stock_ids"] == ["A"]


def test_duplicate_targets_rejected():
    resp = client.post(
        PATH,
        json={
            "stocks": [stock("A", {"NaCl": "1"})],
            "final_volume_ml": "100",
            "targets": [
                {"solute": "NaCl", "concentration": "0.1"},
                {"solute": "NaCl", "concentration": "0.2"},
            ],
        },
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "INVALID_BLEND"
    assert err["details"]["duplicate_targets"] == ["NaCl"]


@pytest.mark.parametrize(
    "payload_mutator",
    [
        lambda b: b["stocks"][0].__setitem__("available_volume_ml", "-1"),
        lambda b: b["stocks"][0]["concentrations"].__setitem__("NaCl", "nan"),
        lambda b: b.__setitem__("final_volume_ml", "Infinity"),
        lambda b: b["targets"][0].__setitem__("concentration", "abc"),
        lambda b: b.__setitem__("final_volume_ml", "0"),
        lambda b: b.__setitem__("final_volume_ml", "-0"),
    ],
)
def test_invalid_decimal_values_rejected(payload_mutator):
    payload = body([stock("A", {"NaCl": "1"})], "100", [("NaCl", "0.1")])
    payload_mutator(payload)
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_BLEND"


@pytest.mark.parametrize("field", ["available_volume_ml", "final_volume_ml"])
def test_json_numbers_rejected(field):
    payload = body([stock("A", {"NaCl": "1"})], "100", [("NaCl", "0.1")])
    if field == "available_volume_ml":
        payload["stocks"][0][field] = 100
    else:
        payload[field] = 100
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_json_number_concentration_rejected():
    payload = body([stock("A", {"NaCl": "1"})], "100", [("NaCl", "0.1")])
    payload["stocks"][0]["concentrations"]["NaCl"] = 1
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_blank_strings_rejected():
    payload = body([stock("  ", {"NaCl": "1"})], "100", [("NaCl", "0.1")])
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_missing_fields_rejected():
    resp = client.post(PATH, json={"stocks": []})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_empty_stocks_and_targets_rejected():
    resp = client.post(
        PATH, json={"stocks": [], "final_volume_ml": "100", "targets": []}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_invalid_decimal_details_pinpoint_field_and_stock():
    payload = body(
        [
            stock("A", {"NaCl": "1"}),
            stock("B", {"NaCl": "bogus"}),
        ],
        "100",
        [("NaCl", "0.1")],
    )
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["invalid_decimals"] == [
        {
            "field": "stocks.concentrations",
            "stock_id": "B",
            "solute": "NaCl",
            "value": "bogus",
        }
    ]


def test_text_fields_are_stripped():
    # 标识、溶质名与数值两侧空白应被裁剪后正常求解
    resp = client.post(
        PATH,
        json={
            "stocks": [
                {
                    "id": "  A  ",
                    "concentrations": {" NaCl ": " 1 "},
                    "available_volume_ml": " 100 ",
                }
            ],
            "final_volume_ml": " 250 ",
            "targets": [{"solute": " NaCl ", "concentration": " 0.1 "}],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["stocks"][0]["id"] == "A"
    assert takes(data) == {"A": "25"}
    assert data["final_concentrations"][0]["solute"] == "NaCl"
