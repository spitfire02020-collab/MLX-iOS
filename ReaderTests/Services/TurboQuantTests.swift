import XCTest
@testable import Reader

// MARK: - TurboQuantScheduler Tests

final class TurboQuantSchedulerTests: XCTestCase {

    // MARK: - Initialization

    func testDefaultWarmupSteps() {
        let scheduler = TurboQuantScheduler()
        XCTAssertEqual(scheduler.warmupSteps, 8, "Default warmup should be 8 steps")
    }

    func testCustomWarmupSteps() {
        let scheduler = TurboQuantScheduler(warmupSteps: 16)
        XCTAssertEqual(scheduler.warmupSteps, 16, "Custom warmup should be 16 steps")
    }

    func testZeroWarmupSteps() {
        let scheduler = TurboQuantScheduler(warmupSteps: 0)
        XCTAssertEqual(scheduler.warmupSteps, 0, "Zero warmup steps should be allowed")
        XCTAssertFalse(scheduler.isInWarmup, "Should not be in warmup when warmupSteps=0")
        XCTAssertTrue(scheduler.shouldQuantize, "Should quantize immediately when warmupSteps=0")
    }

    func testNegativeWarmupStepsClamped() {
        let scheduler = TurboQuantScheduler(warmupSteps: -5)
        XCTAssertEqual(scheduler.warmupSteps, 0, "Negative warmup steps should be clamped to 0")
    }

    func testInitialDecodeStep() {
        let scheduler = TurboQuantScheduler(warmupSteps: 8)
        XCTAssertEqual(scheduler.currentDecodeStep, 0, "Initial decode step should be 0")
    }

    // MARK: - Warmup State

    func testIsInWarmupAtStart() {
        let scheduler = TurboQuantScheduler(warmupSteps: 8)
        XCTAssertTrue(scheduler.isInWarmup, "Should be in warmup at step 0")
    }

    func testIsInWarmupDuringWarmup() {
        var scheduler = TurboQuantScheduler(warmupSteps: 8)
        for _ in 0..<7 {
            XCTAssertTrue(scheduler.isInWarmup, "Should be in warmup during first 7 steps")
            scheduler.advanceStep()
        }
        // At step 7, still in warmup (warmupSteps=8 means steps 0..7)
        XCTAssertTrue(scheduler.isInWarmup, "Step 7 should still be in warmup")
    }

    func testIsNotInWarmupAfterWarmup() {
        var scheduler = TurboQuantScheduler(warmupSteps: 8)
        for _ in 0..<8 {
            scheduler.advanceStep()
        }
        XCTAssertFalse(scheduler.isInWarmup, "Should not be in warmup after 8 steps")
    }

    // MARK: - Should Quantize

    func testShouldNotQuantizeDuringWarmup() {
        let scheduler = TurboQuantScheduler(warmupSteps: 4)
        XCTAssertFalse(scheduler.shouldQuantize, "Should not quantize during warmup")
    }

    func testShouldQuantizeAfterWarmup() {
        var scheduler = TurboQuantScheduler(warmupSteps: 4)
        for _ in 0..<4 {
            scheduler.advanceStep()
        }
        XCTAssertTrue(scheduler.shouldQuantize, "Should quantize after warmup completes")
    }

    func testShouldQuantizeImmediatelyWithZeroWarmup() {
        let scheduler = TurboQuantScheduler(warmupSteps: 0)
        XCTAssertTrue(scheduler.shouldQuantize, "Should quantize immediately with 0 warmup steps")
    }

    // MARK: - Transition Step

    func testTransitionStepAtBoundary() {
        var scheduler = TurboQuantScheduler(warmupSteps: 4)
        for _ in 0..<4 {
            XCTAssertFalse(scheduler.isTransitionStep, "Should not be transition step during warmup")
            scheduler.advanceStep()
        }
        XCTAssertTrue(scheduler.isTransitionStep, "Step 4 should be the transition step")
    }

    func testTransitionStepOnlyOnce() {
        var scheduler = TurboQuantScheduler(warmupSteps: 4)
        for _ in 0..<4 {
            scheduler.advanceStep()
        }
        XCTAssertTrue(scheduler.isTransitionStep, "Step 4 is transition")
        scheduler.advanceStep()
        XCTAssertFalse(scheduler.isTransitionStep, "Step 5 should NOT be transition")
    }

    func testTransitionStepWithZeroWarmup() {
        let scheduler = TurboQuantScheduler(warmupSteps: 0)
        XCTAssertTrue(scheduler.isTransitionStep, "Step 0 should be transition when warmupSteps=0")
    }

    // MARK: - Advance Step

    func testAdvanceStepIncrements() {
        var scheduler = TurboQuantScheduler(warmupSteps: 8)
        XCTAssertEqual(scheduler.currentDecodeStep, 0)
        scheduler.advanceStep()
        XCTAssertEqual(scheduler.currentDecodeStep, 1)
        scheduler.advanceStep()
        XCTAssertEqual(scheduler.currentDecodeStep, 2)
    }

    func testAdvanceStepBeyondWarmup() {
        var scheduler = TurboQuantScheduler(warmupSteps: 2)
        scheduler.advanceStep()
        scheduler.advanceStep()
        scheduler.advanceStep()
        scheduler.advanceStep()
        XCTAssertEqual(scheduler.currentDecodeStep, 4)
        XCTAssertFalse(scheduler.isInWarmup)
        XCTAssertTrue(scheduler.shouldQuantize)
    }

    // MARK: - Reset

    func testResetResetsDecodeStep() {
        var scheduler = TurboQuantScheduler(warmupSteps: 8)
        scheduler.advanceStep()
        scheduler.advanceStep()
        scheduler.advanceStep()
        XCTAssertEqual(scheduler.currentDecodeStep, 3)
        scheduler.reset()
        XCTAssertEqual(scheduler.currentDecodeStep, 0, "Reset should set decode step to 0")
    }

    func testResetRestoresWarmup() {
        var scheduler = TurboQuantScheduler(warmupSteps: 4)
        for _ in 0..<10 {
            scheduler.advanceStep()
        }
        XCTAssertFalse(scheduler.isInWarmup)
        scheduler.reset()
        XCTAssertTrue(scheduler.isInWarmup, "Reset should restore warmup state")
    }

    func testResetPreservesWarmupSteps() {
        var scheduler = TurboQuantScheduler(warmupSteps: 12)
        scheduler.advanceStep()
        scheduler.reset()
        XCTAssertEqual(scheduler.warmupSteps, 12, "Reset should not change warmupSteps")
    }

    // MARK: - Position Counts

    func testWarmupPositionsStoredDuringWarmup() {
        var scheduler = TurboQuantScheduler(warmupSteps: 8)
        XCTAssertEqual(scheduler.warmupPositionsStored, 0, "No warmup positions at step 0")

        scheduler.advanceStep()
        XCTAssertEqual(scheduler.warmupPositionsStored, 1, "1 warmup position at step 1")

        scheduler.advanceStep()
        scheduler.advanceStep()
        XCTAssertEqual(scheduler.warmupPositionsStored, 3, "3 warmup positions at step 3")
    }

    func testWarmupPositionsCappedAtWarmupSteps() {
        var scheduler = TurboQuantScheduler(warmupSteps: 4)
        for _ in 0..<10 {
            scheduler.advanceStep()
        }
        XCTAssertEqual(scheduler.warmupPositionsStored, 4, "Warmup positions should cap at warmupSteps")
    }

    func testQuantizedPositionsDuringWarmup() {
        let scheduler = TurboQuantScheduler(warmupSteps: 8)
        XCTAssertEqual(scheduler.quantizedPositionsStored, 0, "No quantized positions during warmup")
    }

    func testQuantizedPositionsAfterWarmup() {
        var scheduler = TurboQuantScheduler(warmupSteps: 4)
        for _ in 0..<7 {
            scheduler.advanceStep()
        }
        XCTAssertEqual(scheduler.quantizedPositionsStored, 3, "Should have 3 quantized positions (steps 4, 5, 6)")
    }

    func testPositionCountsConsistency() {
        var scheduler = TurboQuantScheduler(warmupSteps: 5)
        for i in 0..<20 {
            let total = scheduler.warmupPositionsStored + scheduler.quantizedPositionsStored
            XCTAssertEqual(total, i, "Total positions should equal current step at step \(i)")
            scheduler.advanceStep()
        }
    }

    // MARK: - Full Lifecycle

    func testFullLifecycle() {
        var scheduler = TurboQuantScheduler(warmupSteps: 3)

        // Step 0: warmup
        XCTAssertTrue(scheduler.isInWarmup)
        XCTAssertFalse(scheduler.shouldQuantize)
        XCTAssertFalse(scheduler.isTransitionStep)
        XCTAssertEqual(scheduler.currentDecodeStep, 0)
        scheduler.advanceStep()

        // Step 1: warmup
        XCTAssertTrue(scheduler.isInWarmup)
        XCTAssertFalse(scheduler.shouldQuantize)
        XCTAssertFalse(scheduler.isTransitionStep)
        scheduler.advanceStep()

        // Step 2: warmup
        XCTAssertTrue(scheduler.isInWarmup)
        XCTAssertFalse(scheduler.shouldQuantize)
        XCTAssertFalse(scheduler.isTransitionStep)
        scheduler.advanceStep()

        // Step 3: transition (first quantized step)
        XCTAssertFalse(scheduler.isInWarmup)
        XCTAssertTrue(scheduler.shouldQuantize)
        XCTAssertTrue(scheduler.isTransitionStep)
        XCTAssertEqual(scheduler.warmupPositionsStored, 3)
        XCTAssertEqual(scheduler.quantizedPositionsStored, 0)
        scheduler.advanceStep()

        // Step 4: post-warmup quantized
        XCTAssertFalse(scheduler.isInWarmup)
        XCTAssertTrue(scheduler.shouldQuantize)
        XCTAssertFalse(scheduler.isTransitionStep)
        XCTAssertEqual(scheduler.warmupPositionsStored, 3)
        XCTAssertEqual(scheduler.quantizedPositionsStored, 1)
        scheduler.advanceStep()

        // Step 5: still quantized
        XCTAssertEqual(scheduler.quantizedPositionsStored, 2)

        // Reset and verify
        scheduler.reset()
        XCTAssertTrue(scheduler.isInWarmup)
        XCTAssertEqual(scheduler.currentDecodeStep, 0)
        XCTAssertEqual(scheduler.warmupPositionsStored, 0)
        XCTAssertEqual(scheduler.quantizedPositionsStored, 0)
    }

    func testSingleWarmupStep() {
        var scheduler = TurboQuantScheduler(warmupSteps: 1)

        // Step 0: warmup
        XCTAssertTrue(scheduler.isInWarmup)
        XCTAssertFalse(scheduler.isTransitionStep)
        scheduler.advanceStep()

        // Step 1: transition
        XCTAssertFalse(scheduler.isInWarmup)
        XCTAssertTrue(scheduler.isTransitionStep)
        scheduler.advanceStep()

        // Step 2: quantized
        XCTAssertFalse(scheduler.isInWarmup)
        XCTAssertFalse(scheduler.isTransitionStep)
        XCTAssertTrue(scheduler.shouldQuantize)
    }

    func testLargeWarmupSteps() {
        var scheduler = TurboQuantScheduler(warmupSteps: 1000)
        for i in 0..<999 {
            XCTAssertTrue(scheduler.isInWarmup, "Should be in warmup at step \(i)")
            scheduler.advanceStep()
        }
        XCTAssertTrue(scheduler.isInWarmup, "Should be in warmup at step 999")
        scheduler.advanceStep()
        XCTAssertFalse(scheduler.isInWarmup, "Should NOT be in warmup at step 1000")
        XCTAssertTrue(scheduler.isTransitionStep, "Step 1000 should be transition")
    }

    // MARK: - Multiple Resets

    func testMultipleResets() {
        var scheduler = TurboQuantScheduler(warmupSteps: 3)

        // First run
        for _ in 0..<5 {
            scheduler.advanceStep()
        }
        XCTAssertEqual(scheduler.currentDecodeStep, 5)

        // Reset and run again
        scheduler.reset()
        XCTAssertEqual(scheduler.currentDecodeStep, 0)
        XCTAssertTrue(scheduler.isInWarmup)

        for _ in 0..<3 {
            scheduler.advanceStep()
        }
        XCTAssertTrue(scheduler.isTransitionStep)

        // Reset again
        scheduler.reset()
        XCTAssertTrue(scheduler.isInWarmup)
        XCTAssertEqual(scheduler.currentDecodeStep, 0)
    }
}

// MARK: - ChatterboxConfig TurboQuant Tests

final class ChatterboxConfigTurboQuantTests: XCTestCase {

    func testDefaultConfigHasWarmupSteps() {
        let config = ChatterboxConfig.default
        XCTAssertEqual(config.turboQuantWarmupSteps, 8, "Default warmup should be 8 steps")
    }

    func testDefaultConfigHasCorrectHeadDim() {
        let config = ChatterboxConfig.default
        XCTAssertEqual(config.headDim, 64, "Head dimension should be 64")
    }

    func testDefaultConfigHasCorrectNumKVHeads() {
        let config = ChatterboxConfig.default
        XCTAssertEqual(config.numKVHeads, 16, "Number of KV heads should be 16")
    }
}
