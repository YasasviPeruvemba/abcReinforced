module adder (a0, b0, c0, s0, s1);

input a0, b0, c0;
output s0, s1;

assign s0 = a0 ^ b0 ^ c0 ;
assign s1 = (a0 & b0) | (b0 & c0) | (c0 & a0);

endmodule

