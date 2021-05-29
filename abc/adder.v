module rca2 (a0, b0, a1, b1, s0, s1, s2);
input a0, b0, a1, b1;
output s0, s1, s2;
wire c0;

assign s0 = a0 ^ b0 ;
assign c0 = a0 & b0 ;
assign s1 = a1 ^ b1 ^ c0;
assign s2 = (a1 & b1) | (c0 & (a1 ^ b1));

endmodule
