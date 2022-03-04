using RandomizedSDP
using LinearAlgebra
using Test

const RS = RandomizedSDP

@testset "RandomizedSDP.jl" begin
    n = 100
    for i in 1:5
        b, F, G, xstar, _ = RS.generate_random_sdp(n; rand_seed=i)
        pstar = dot(b, xstar)

        ## Real Problem
        c = Vector(RS.vec_symm(-G))
        A = RS.build_A(F)
        sdp = RS.SDP(c, A, b)
        result = RS.solve!(
            sdp; 
            relax=false, 
            print_iter=10, 
            indirect=true, 
            precondition=true, 
            logging=true, 
            verbose=false
        )
        @test abs(sdp.obj_val - pstar) / min(sdp.obj_val, pstar) <= 1e-6
    end
end
