#include "decode_macros.h"
#include "disasm.h"
#include "extension.h"
#include "processor.h"

#include <cstdint>

static constexpr uint32_t xqmac8_match =
    (0x2bU << 25) | (0x0U << 12) | 0x0bU;
static constexpr uint32_t xqmac8_mask =
    (0x7fU << 25) | (0x7U << 12) | 0x7fU;

static constexpr uint32_t xrequant_match =
    (0x2bU << 25) | (0x1U << 12) | 0x0bU;
static constexpr uint32_t xrequant_mask =
    (0x7fU << 25) | (0x7U << 12) | 0x7fU;

struct xreg_arg_t : public arg_t {
    explicit xreg_arg_t(int which) : which(which) {}
    std::string to_string(insn_t insn) const override
    {
        switch (which) {
        case 0: return xpr_name[insn.rd()];
        case 1: return xpr_name[insn.rs1()];
        default: return xpr_name[insn.rs2()];
        }
    }
    int which;
};

static xreg_arg_t x_rd(0);
static xreg_arg_t x_rs1(1);
static xreg_arg_t x_rs2(2);

static reg_t xqmac8_execute(processor_t* processor, insn_t insn, reg_t pc)
{
    state_t* state = processor->get_state();
    reg_t packed_a = state->XPR[insn.rs1()];
    reg_t packed_b = state->XPR[insn.rs2()];
    int64_t accumulator = static_cast<int64_t>(state->XPR[insn.rd()]);
    int64_t result = accumulator;

    for (int lane = 0; lane < 8; ++lane) {
        int shift = lane * 8;
        int8_t a = static_cast<int8_t>(packed_a >> shift);
        int8_t b = static_cast<int8_t>(packed_b >> shift);
        result += static_cast<int64_t>(a) * static_cast<int64_t>(b);
    }

    if (insn.rd() != 0) {
        state->XPR.write(insn.rd(), static_cast<reg_t>(result));
    }
    return pc + 4;
}

static reg_t xrequant_execute(processor_t* processor, insn_t insn, reg_t pc)
{
    state_t* state = processor->get_state();
    reg_t ctrl = state->XPR[insn.rd()];
    int32_t zero_point = static_cast<int8_t>(ctrl & 0xff);
    int shift = static_cast<int>((ctrl >> 8) & 0xff);
    int32_t accumulator = static_cast<int32_t>(state->XPR[insn.rs1()]);
    int32_t multiplier = static_cast<int32_t>(state->XPR[insn.rs2()]);

    int64_t product =
        static_cast<int64_t>(accumulator) * static_cast<int64_t>(multiplier);
    if (shift > 0) {
        const int64_t rounding = static_cast<int64_t>(1) << (shift - 1);
        product = (product + rounding) >> shift;
    }
    product += zero_point;
    if (product > 127) {
        product = 127;
    } else if (product < -128) {
        product = -128;
    }

    if (insn.rd() != 0) {
        state->XPR.write(insn.rd(), static_cast<reg_t>(static_cast<int8_t>(product)));
    }
    return pc + 4;
}

struct xqmac8_extension_t : public extension_t {
    const char* name() const override { return "xqmac8"; }

    std::vector<insn_desc_t> get_instructions(const processor_t&) override
    {
        return {
            {xqmac8_match, xqmac8_mask, xqmac8_execute, xqmac8_execute,
             xqmac8_execute, xqmac8_execute, xqmac8_execute, xqmac8_execute,
             xqmac8_execute, xqmac8_execute},
            {xrequant_match, xrequant_mask, xrequant_execute, xrequant_execute,
             xrequant_execute, xrequant_execute, xrequant_execute,
             xrequant_execute, xrequant_execute, xrequant_execute},
        };
    }

    std::vector<disasm_insn_t*> get_disasms(const processor_t*) override
    {
        return {
            new disasm_insn_t("xqmac8", xqmac8_match, xqmac8_mask,
                              {&x_rd, &x_rs1, &x_rs2}),
            new disasm_insn_t("xrequant", xrequant_match, xrequant_mask,
                              {&x_rd, &x_rs1, &x_rs2}),
        };
    }
};

REGISTER_EXTENSION(xqmac8, []() {
    static xqmac8_extension_t extension;
    return &extension;
})
